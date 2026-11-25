"""Reuse frozen encodings of real audio without changing inference architecture.

Cached tokens must come from normalized real windows and the SAME frozen audio
backbone/projection used by every compared checkpoint. Normal deployment still
uses the unchanged model's encode_audio/forward_sequence implementations.

The temporary method override is intentionally local to a synchronous forward;
concurrent forwards on the same model instance are unsupported.
"""
from __future__ import annotations

import numpy as np
import torch


def freeze_audio(model):
    """Freeze audio weights and their running statistics; return the model.

    Call again after model.train() if any direct encoder calls are made during
    training: PyTorch's recursive train() also changes frozen modules' modes.
    Cached forwards never execute these modules, so their statistics stay fixed.
    """
    for name in ('audio_backbone', 'audio_projection'):
        module = getattr(model, name, None)
        if module is None:
            raise ValueError('Cached real audio requires a multimodal audio backbone/projection')
        module.requires_grad_(False)
        module.eval()
    return model


def _token_dimensions(model):
    config = getattr(model, 'config', None)
    return int(getattr(config, 'audio_frames', 28)), int(getattr(config, 'd_model', 128))


def prepare_audio_tokens(model, data, device, batch_size=64):
    """Add float32 [N,T,28,d_model] tokens from data['audio'] real windows.

    data['audio'] must already use the clean reference checkpoint's audio mean
    and standard deviation, shaped [N,T,6,64,28]. Only bounded minibatches move
    to the requested device. Neither input audio nor model weights are changed.
    The returned value is also assigned to data['audio_tokens'].
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    audio = np.asarray(data['audio'])
    if audio.ndim != 5 or audio.shape[1] < 1 or audio.shape[2:] != (6, 64, 28):
        raise ValueError('Normalized real audio must have shape [N,T,6,64,28]')
    if audio.dtype.kind != 'f':
        raise ValueError('Normalized real audio must be floating point')
    freeze_audio(model)
    frames, width = _token_dimensions(model)
    windows = audio.reshape(-1, 6, 64, 28)
    encoded = np.empty((len(windows), frames, width), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            chunk = np.asarray(windows[start:start + batch_size], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError('Normalized audio contains nonfinite samples')
            # Copy only a bounded window batch if the source array is read-only.
            if not chunk.flags.writeable:
                chunk = chunk.copy()
            inputs = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
            result = model.encode_audio(inputs)
            if result.shape != (len(chunk), frames, width):
                raise ValueError('Audio encoder returned unexpected token dimensions')
            values = result.detach().to(device='cpu', dtype=torch.float32).numpy()
            if not np.isfinite(values).all():
                raise ValueError('Audio encoder returned nonfinite tokens')
            encoded[start:start + len(chunk)] = values
    data['audio_tokens'] = encoded.reshape(audio.shape[0], audio.shape[1], frames, width)
    return data['audio_tokens']


def forward_cached(model, image, audio_tokens, hidden=None, *, sequence_length=None, **timing):
    """Run existing forward_sequence, replaying one frozen token set per event.

    Fusion, vision and GRU operations retain gradients. Tokens are detached;
    audio weights are frozen. Existing shape checks still see [B,T,6,64,28]
    audio via a zero-stride placeholder, never fabricated encoded audio. The
    placeholder cannot reach an encoder: every encode_audio call is intercepted.
    Original method lookup is restored even when the model raises an exception.
    """
    if not isinstance(audio_tokens, torch.Tensor) or audio_tokens.ndim != 4:
        raise ValueError('audio_tokens must be a tensor [B,T,28,d_model]')
    batch, length = audio_tokens.shape[:2]
    if batch < 1 or length < 1 or audio_tokens.shape[2:] != _token_dimensions(model):
        raise ValueError('audio_tokens dimensions must match the model and contain at least one event')
    if sequence_length is not None and sequence_length != length:
        raise ValueError('sequence_length must match cached audio event count')
    if not audio_tokens.is_floating_point():
        raise ValueError('audio_tokens must be floating point')
    if image is not None and image.shape[0] != batch:
        raise ValueError('Image and cached audio batch sizes must agree')
    current_encoder = model.encode_audio
    if getattr(current_encoder, '_cached_audio_replay', False):
        raise RuntimeError('Nested/concurrent cached forwards on one model are unsupported')
    freeze_audio(model)
    tokens = audio_tokens.detach()
    event = 0

    def replay(_audio):
        nonlocal event
        if event >= length:
            raise RuntimeError('forward_sequence requested more audio events than cached')
        result = tokens[:, event]
        event += 1
        return result

    replay._cached_audio_replay = True
    sentinel = object()
    previous_override = model.__dict__.get('encode_audio', sentinel)
    model.encode_audio = replay
    try:
        # Expanded views allocate only one scalar, while satisfying each model's
        # raw audio interface. The encoder implementation is never executed.
        placeholder = tokens.new_zeros(1, 1, 1, 1, 1).expand(batch, length, 6, 64, 28)
        result = model.forward_sequence(image, placeholder, hidden,
                                        sequence_length=length, **timing)
        if event != length:
            raise RuntimeError('forward_sequence did not consume every cached audio event')
        return result
    finally:
        if previous_override is sentinel:
            delattr(model, 'encode_audio')
        else:
            model.encode_audio = previous_override

"""CPU-only behavioral checks for reusable real-audio token encodings."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from image_exposure_experiments.audio_cache import freeze_audio, forward_cached, prepare_audio_tokens


class TinyMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(audio_frames=28, d_model=8)
        self.audio_backbone = nn.Sequential(nn.Flatten(), nn.Linear(6 * 64 * 28, 12),
                                             nn.BatchNorm1d(12), nn.ReLU())
        self.audio_projection = nn.Linear(12, 28 * 8)
        self.visual = nn.Linear(3 * 4 * 4, 8)
        self.gru = nn.GRU(8, 12, batch_first=True)
        self.head = nn.Linear(12, 3)
        self.encoder_calls = 0
        self.raise_after_audio = False

    def encode_audio(self, audio):
        self.encoder_calls += 1
        return self.audio_projection(self.audio_backbone(audio)).reshape(len(audio), 28, 8)

    def forward_sequence(self, image, audio, hidden=None, *, sequence_length=None, **timing):
        assert audio.shape[2:] == (6, 64, 28)
        assert sequence_length == audio.shape[1]
        visual = self.visual(image.flatten(1))
        steps = []
        for index in range(sequence_length):
            encoded = self.encode_audio(audio[:, index])
            if self.raise_after_audio:
                raise RuntimeError('intentional model failure')
            steps.append(visual + encoded.mean(1))
        states, final = self.gru(torch.stack(steps, dim=1), hidden)
        return {'position': self.head(states), 'hidden': final}


def example():
    torch.manual_seed(32)
    model = TinyMultimodal().eval()
    images = torch.randn(2, 3, 4, 4)
    audio = torch.randn(2, 3, 6, 64, 28)
    return model, images, audio


def test_real_encoder_and_cached_forward_match_and_restore_method():
    model, images, audio = example()
    original = model.encode_audio.__func__
    expected = model.forward_sequence(images, audio, sequence_length=3)
    data = {'audio': audio.numpy()}
    tokens = prepare_audio_tokens(model, data, 'cpu', batch_size=2)
    before = model.encoder_calls
    actual = forward_cached(model, images, torch.from_numpy(tokens))
    torch.testing.assert_close(actual['position'], expected['position'], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual['hidden'], expected['hidden'], atol=1e-7, rtol=1e-6)
    assert model.encoder_calls == before
    assert model.encode_audio.__func__ is original
    assert 'encode_audio' not in model.__dict__
    assert data['audio_tokens'].dtype == np.float32


def test_audio_weights_and_statistics_are_frozen_but_visual_gradients_flow():
    model, images, audio = example()
    original_weights = {key: value.clone() for key, value in model.audio_backbone.state_dict().items()}
    tokens = prepare_audio_tokens(model, {'audio': audio.numpy()}, 'cpu', batch_size=2)
    model.train()
    result = forward_cached(model, images, torch.from_numpy(tokens))
    result['position'].square().mean().backward()
    assert model.visual.weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    assert not model.audio_backbone.training and not model.audio_projection.training
    assert all(p.grad is None and not p.requires_grad for p in model.audio_backbone.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.audio_projection.parameters())
    for key, value in model.audio_backbone.state_dict().items():
        torch.testing.assert_close(value, original_weights[key], atol=0, rtol=0)


def test_exception_restores_class_method_and_previous_instance_override():
    model, images, audio = example()
    tokens = prepare_audio_tokens(model, {'audio': audio.numpy()}, 'cpu')
    previous = model.encode_audio
    model.encode_audio = previous
    model.raise_after_audio = True
    with pytest.raises(RuntimeError, match='intentional model failure'):
        forward_cached(model, images, torch.from_numpy(tokens))
    assert model.__dict__['encode_audio'] is previous
    assert model.encode_audio is previous


def test_freeze_preserves_unrelated_training_and_parameters():
    model, _, _ = example()
    model.train()
    freeze_audio(model)
    assert model.training and model.visual.training
    assert model.visual.weight.requires_grad
    assert not model.audio_backbone.training


def test_incorrect_token_dimensions_are_rejected_without_override():
    model, images, _ = example()
    with pytest.raises(ValueError):
        forward_cached(model, images, torch.zeros(2, 3, 20, 8))
    assert 'encode_audio' not in model.__dict__


def test_real_audio_order_is_preserved_across_cache_minibatches():
    model, _, audio = example()
    cached = prepare_audio_tokens(model, {'audio': audio.numpy()}, 'cpu', batch_size=4)
    with torch.inference_mode():
        for frame in range(2):
            for event in range(3):
                single = model.encode_audio(audio[frame:frame + 1, event])
                # CPU GEMM kernels accumulate singleton/batched linear layers
                # in slightly different orders; ordering errors are far larger.
                torch.testing.assert_close(torch.from_numpy(cached[frame, event]), single[0],
                                           atol=1e-6, rtol=1e-5)

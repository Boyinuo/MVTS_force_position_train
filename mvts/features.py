"""Event-aligned six-channel ADC windows and log-mel power features."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn


@dataclass
class FeatureConfig:
    sample_rate: float = 50000.0
    n_fft: int = 512
    hop_length: int = 128
    n_mels: int = 64
    samples: int | None = None
    adc_scale: float = 65535.0
    alignment_mode: str | None = None
    before_samples: int | None = None
    after_samples: int | None = None
    stft_right_align: bool = False

    def __post_init__(self):
        # Defaults and older checkpoints/callers use equal pre/post counts.
        # New callers pass alignment_mode and counts from their model configuration.
        if all(value is None for value in (
                self.alignment_mode, self.before_samples, self.after_samples)):
            if self.samples is None:
                self.samples = 2000
            if type(self.samples) is not int or self.samples < 2 or self.samples % 2:
                raise ValueError("Legacy event audio samples must be a positive even number")
            self.alignment_mode = "before_after"
            self.before_samples = self.after_samples = self.samples // 2

        if self.alignment_mode not in ("before_after", "before_only"):
            raise ValueError("alignment_mode must be 'before_after' or 'before_only'")
        if self.alignment_mode == "before_only":
            self.after_samples = 0
        for name in ("before_samples", "after_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        total = self.before_samples + self.after_samples
        if total < 1:
            raise ValueError("Event audio window must contain at least one sample")
        if self.samples is not None and (type(self.samples) is not int or self.samples != total):
            raise ValueError("samples must equal the selected before/after sample counts")
        self.samples = total

    @property
    def audio_frames(self):
        return (self.samples - self.n_fft) // self.hop_length + 1

    def to_dict(self):
        return asdict(self)


def extract_event_audio(data, samples=None, *, config=None):
    """Select configured pre/post-event ADC samples from an open NPZ or mapping.

    The event boundary is the exported audio_before count. Legacy exports without
    window metadata use the midpoint. before_only ends just before the boundary.
    Passing samples alone retains the legacy equal pre/post window.
    """
    if config is not None and samples is not None:
        raise ValueError("Pass either config or samples")
    config = config or FeatureConfig(samples=samples)
    adc = data["adc"]
    if adc.ndim != 2 or adc.shape[1] != 6 or not np.isfinite(adc).all():
        raise ValueError(f"Invalid ADC: {adc.shape}")
    counts = {}
    for name in ("audio_before", "audio_after"):
        if name in data:
            value = np.asarray(data[name])
            if value.ndim != 0 or value.dtype.kind not in "iu" or value.item() < 0:
                raise ValueError(f"Invalid {name}: expected a non-negative integer scalar")
            counts[name] = int(value.item())
    if counts:
        before = counts.get("audio_before", len(adc) - counts.get("audio_after", 0))
        after = counts.get("audio_after", len(adc) - before)
        if before < 0 or after < 0 or before + after != len(adc):
            raise ValueError("Audio window metadata does not match ADC length")
    else:
        if len(adc) % 2:
            raise ValueError("Cannot infer event boundary from odd ADC length")
        before, after = len(adc) // 2, len(adc) // 2
    if before < config.before_samples or after < config.after_samples:
        raise ValueError(f"Event audio needs at least {config.before_samples} samples before "
                         f"and {config.after_samples} samples after the event")
    return adc[before - config.before_samples:before + config.after_samples]


class LogMel(nn.Module):
    """Slaney mel scale and area-normalized filters, log10(power + 1e-10).

    Remove per-channel DC, but preserve relative channel/sample amplitudes.
    STFT center=False gives floor((samples - n_fft)/hop_length) + 1 frames.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FeatureConfig()
        c = self.config
        if c.samples < c.n_fft:
            raise ValueError(f"Audio window needs at least n_fft={c.n_fft} samples for Log-Mel")
        self.register_buffer("window", torch.hann_window(c.n_fft))
        # Slaney's linear low-frequency region avoids empty low mel bands for
        # 64 filters / 512 FFT bins at 48-50 kHz (HTK would have empty bands).
        logstep = np.log(6.4) / 27
        nyquist = c.sample_rate / 2
        max_mel = nyquist / (200 / 3) if nyquist < 1000 else 15 + np.log(nyquist / 1000) / logstep
        mel = torch.linspace(0, float(max_mel), c.n_mels + 2)
        hz = torch.where(mel < 15, mel * (200 / 3), 1000 * torch.exp((mel - 15) * logstep))
        frequencies = torch.linspace(0, c.sample_rate / 2, c.n_fft // 2 + 1)
        ascending = (frequencies[:, None] - hz[:-2]) / (hz[1:-1] - hz[:-2])
        descending = (hz[2:] - frequencies[:, None]) / (hz[2:] - hz[1:-1])
        bank = torch.minimum(ascending, descending).clamp_min(0).T.contiguous()
        bank = bank * (2 / (hz[2:] - hz[:-2]))[:, None]
        if (bank.sum(dim=1) == 0).any():
            raise ValueError("Empty mel filter; reduce n_mels or increase n_fft")
        self.register_buffer("mel_bank", bank)

    def forward(self, adc):
        if adc.shape[-2:] != (self.config.samples, 6):
            raise ValueError(f"Expected ADC [...,{self.config.samples},6], got {tuple(adc.shape)}")
        signal = adc.float().transpose(-1, -2) / self.config.adc_scale
        signal = signal - signal.mean(dim=-1, keepdim=True)
        if self.config.stft_right_align:
            span = self.config.n_fft + (self.config.audio_frames - 1) * self.config.hop_length
            signal = signal[..., -span:]
        leading = signal.shape[:-1]
        spectrum = torch.stft(signal.reshape(-1, signal.shape[-1]),
                              n_fft=self.config.n_fft, hop_length=self.config.hop_length,
                              window=self.window, center=False, return_complex=True)
        mel = self.mel_bank @ spectrum.abs().square()
        return torch.log10(mel.clamp_min(1e-10)).reshape(*leading, self.config.n_mels, -1)


def load_image(path):
    # Keep the complete sensor field; no random crop/flip changing coordinate labels.
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                       dtype=np.float32, copy=True)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1) / 255.0
    return (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225])[:, None, None]

"""Behavioral checks for Scheme II's causal gating and physical timing."""
import numpy as np
import pytest
import torch

from mvts.architectures import available_models, build_model
from mvts.dual_rate_data import event_timing_seconds


def normalization():
    return {"target_mean": [0.] * 9 + [1., -2.],
            "target_std": [1.] * 9 + [2., 3.]}


def new_model(**config):
    model = build_model("model_slip_gated", {"dropout": 0, **config}).eval()
    model.set_normalization(normalization())
    return model


def test_sequence_matches_streaming_with_real_timing_and_delayed_history():
    torch.manual_seed(13)
    assert "model_slip_gated" in available_models()
    model = new_model(disabled_mic_channels=[5])
    image, audio = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 6, 64, 28)
    dt = torch.tensor([[.006, .008, .007]])
    age = torch.tensor([[.002, .010, .017]])
    with torch.no_grad():
        expected = model.forward_sequence(image, audio, delta_t=dt, image_age=age)
        zero_reset = model.forward_sequence(image, audio, torch.zeros(1, 1, 256),
            delta_t=dt, image_age=age, has_history=torch.tensor([False]))
    calls = []
    hook = model.vision_backbone.register_forward_hook(lambda *args: calls.append(1))
    observed = [model.step(audio[:, i], image=image if i == 0 else None, frame_id="frame",
                           delta_t=dt[:, i], image_age=age[:, i]) for i in range(3)]
    assert len(calls) == 1
    keys = ("position", "wrench", "slip", "slip_prior", "visual_weight", "audio_weight", "prior_speed_mm_s")
    for key in keys:
        torch.testing.assert_close(expected[key], zero_reset[key])
        torch.testing.assert_close(expected[key], torch.stack([r[key] for r in observed], dim=1))
    torch.testing.assert_close(expected["hidden"], observed[-1]["hidden"])
    torch.testing.assert_close(expected["visual_weight"] + expected["audio_weight"], torch.ones(1, 3))
    model.step(audio[:, 0], image=image, frame_id="new", delta_t=.005)
    assert model._stream_image_age.item() == 0
    model.step(audio[:, 1], delta_t=.009)
    assert model._stream_image_age.item() == pytest.approx(.009)
    assert not any("stream" in key for key in model.state_dict())
    model.reset_state()
    with pytest.raises(ValueError, match="image"):
        model.step(audio[:, 0])
    hook.remove()


def test_future_audio_cannot_change_earlier_predictions_or_gates():
    model = new_model()
    image, audio = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 6, 64, 28)
    changed = audio.clone()
    changed[:, 2] += 5.
    with torch.no_grad():
        a = model.forward_sequence(image, audio)
        b = model.forward_sequence(image, changed)
    for key in ("position", "wrench", "slip", "slip_prior", "visual_weight"):
        torch.testing.assert_close(a[key][:, :2], b[key][:, :2])


def test_gate_uses_physical_speed_and_age_and_can_detach_only_the_prior_path():
    model = new_model()
    with torch.no_grad():
        model.slip_prior_head[-1].weight.zero_()
        model.slip_prior_head[-1].bias.copy_(torch.tensor([3., 4.]))
        # Controlled nonconstant gate: verify that changing dt / age can actually
        # affect fusion weights, irrespective of random network initialization.
        model.modality_gate[0].weight.zero_()
        model.modality_gate[0].bias.zero_()
        model.modality_gate[0].weight[0, -2:] = torch.tensor([1., 1.])
        model.modality_gate[-1].weight.zero_()
        model.modality_gate[-1].weight[0, 0] = 1.
        model.modality_gate[-1].bias.zero_()
    image, audio = torch.randn(1, 3, 64, 64), torch.randn(1, 6, 64, 28)
    visual = model.encode_visual(image).detach()
    a = model.fuse(visual, audio, delta_t=.02, image_age=0.)
    b = model.fuse(visual, audio, delta_t=.04, image_age=0.)
    c = model.fuse(visual, audio, delta_t=.04, image_age=.03)
    # [3,4] normalized -> [7,10] mm, NOT the normalized magnitude of 5.
    torch.testing.assert_close(a["slip_prior_mm"], torch.tensor([[7., 10.]]))
    assert a["prior_speed_mm_s"].item() == pytest.approx(np.sqrt(149) / .02)
    assert a["visual_weight"].item() > b["visual_weight"].item()
    assert c["visual_weight"].item() > b["visual_weight"].item()
    a["visual_weight"].sum().backward()
    assert all(p.grad is None for p in model.slip_prior_head.parameters())
    model.zero_grad(set_to_none=True)
    model.config.detach_slip_for_gate = False
    d = model.fuse(visual, audio, delta_t=.02)
    d["visual_weight"].sum().backward()
    assert model.slip_prior_head[-1].bias.grad.abs().sum() > 0


def test_auxiliary_supervision_and_final_tasks_train_all_branches():
    model = new_model()
    result = model.forward_sequence(torch.randn(1, 3, 64, 64), torch.randn(1, 2, 6, 64, 28))
    sum(result[k].square().mean() for k in ("position", "wrench", "slip", "slip_prior")).backward()
    for parameter in (model.vision_backbone.stem[0].weight, model.audio_backbone.stem[0].weight,
                      model.visual_attention.in_proj_weight, model.audio_attention.in_proj_weight,
                      model.query.weight, model.initial_query, model.gru.weight_hh_l0,
                      model.slip_prior_head[-1].weight, model.modality_gate[0].weight,
                      model.modality_gate[-1].weight, model.position_head[-1].weight,
                      model.force_head[-1].weight, model.slip_head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_nan_in_disabled_microphone_never_reaches_new_model():
    model = new_model(disabled_mic_channels=[5])
    image, audio = torch.randn(1, 3, 64, 64), torch.randn(1, 6, 64, 28)
    changed = audio.clone()
    changed[:, 4] = float("nan")
    with torch.no_grad():
        a, b = model(image, audio), model(image, changed)
    for key in ("position", "wrench", "slip", "slip_prior", "visual_weight"):
        torch.testing.assert_close(a[key], b[key])


def test_timestamp_precision_and_independent_press_initialization():
    base = 700_000_000_000_000_000
    times = np.array([base, base + 7_000_001], dtype=np.int64)
    previous = np.array([base - 6_000_003, base], dtype=np.int64)
    timing = event_timing_seconds(times, previous, base, 2)
    np.testing.assert_allclose(timing["delta_t"], [.006000003, .007000001], rtol=1e-7)
    np.testing.assert_allclose(timing["image_age"], [0., .007000001], rtol=1e-7)
    press = event_timing_seconds(times[:1], None, base + 1_000_000, 1, independent=True)
    assert press["delta_t"].item() == pytest.approx(1 / 150)
    assert press["image_age"].item() == 0
    with pytest.raises(ValueError, match="previous_pc_ns"):
        event_timing_seconds(times, None, base, 2)
    with pytest.raises(ValueError, match="causal"):
        event_timing_seconds(times, times, base, 2)


def test_invalid_normalization_timing_and_configuration_are_rejected():
    model = build_model("model_slip_gated").eval()
    image, audio = torch.randn(1, 3, 64, 64), torch.randn(1, 6, 64, 28)
    with pytest.raises(RuntimeError, match="normalization"):
        model(image, audio)
    model.set_normalization(normalization())
    for dt in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="delta_t"):
            model(image, audio, delta_t=dt)
    with pytest.raises(ValueError, match="image_age"):
        model(image, audio, image_age=-.1)
    with pytest.raises(ValueError, match="not model inputs"):
        model(image, audio, torch.tensor([1]))
    with pytest.raises(ValueError, match="image_only"):
        build_model("model_slip_gated", {"image_only": True})

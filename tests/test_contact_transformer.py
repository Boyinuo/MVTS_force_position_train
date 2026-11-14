import numpy as np
import pytest
import torch

from mvts.architectures import available_models, build_model
from mvts.engine import load_checkpoint
from mvts.features import FeatureConfig, extract_event_audio
from mvts.model_contact_concat_sensor_final import ModelConfig as ConcatModelConfig
from mvts.model_contact_transformer_sensor_final import ModelConfig


MODEL_NAME = "model_contact_transformer_sensor_final"


def test_contact_changes_transformer_inputs_and_receives_gradients():
    torch.manual_seed(19)
    model = build_model(MODEL_NAME, {"dropout": 0}).eval()
    image = torch.randn(1, 3, 224, 224).repeat(2, 1, 1, 1)
    audio = torch.randn(1, 6, 64, 12).repeat(2, 1, 1, 1)
    observed = {}

    def capture(name):
        def hook(_module, args):
            observed[name] = args[0].detach().clone()
        return hook

    hooks = [module.register_forward_pre_hook(capture(name)) for name, module in (
        ("transformer", model.transformer), ("position", model.position_head), ("wrench", model.force_head))]
    result = model(image, audio, torch.tensor([0, 1]))
    for hook in hooks:
        hook.remove()
    assert observed["transformer"].shape == (2, 13, 256)
    expected_difference = (model.contact_embedding.weight[1] - model.contact_embedding.weight[0]).detach()
    torch.testing.assert_close(observed["transformer"][1] - observed["transformer"][0],
                                expected_difference.expand(13, -1), rtol=1e-5, atol=1e-5)
    assert observed["position"].shape == observed["wrench"].shape == (2, 256)
    assert result["position"].shape == (2, 3)
    assert result["wrench"].shape == (2, 6)
    for value in result.values():
        assert not torch.allclose(value[0], value[1])
    (result["position"].square().mean() + result["wrench"].square().mean()).backward()
    for weight in (model.contact_embedding.weight, model.vision_backbone.stem[0].weight,
                   model.audio_backbone.stem[0].weight):
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert (model.contact_embedding.weight.grad.abs().sum(dim=1) > 0).all()


def test_contact_validation_before_encoding():
    model = build_model(MODEL_NAME).eval()
    image, audio = torch.zeros(2, 3, 224, 224), torch.zeros(2, 6, 64, 12)
    for contact in (None, torch.tensor(1), torch.tensor([0]), torch.tensor([[0, 1]]),
                    torch.tensor([0.5, 1]), torch.tensor([-1, 1]), torch.tensor([0, 2]),
                    torch.tensor([float("nan"), 1]), torch.tensor([0, float("inf")])):
        with pytest.raises(ValueError, match="starts_in_contact"):
            model.encode_tokens(image, audio, contact)
    with torch.inference_mode():
        integer = model.encode_tokens(image, audio, torch.tensor([0, 1]))
        boolean = model.encode_tokens(image, audio, torch.tensor([[False], [True]]))
    torch.testing.assert_close(integer, boolean)
    with pytest.raises(ValueError, match="use_contact"):
        build_model(MODEL_NAME, {"use_contact": False})


def test_alignment_and_checkpoint_match_configured_contact_model(tmp_path):
    assert MODEL_NAME in available_models()
    config, previous = ModelConfig(), ConcatModelConfig()
    assert config.to_dict() == previous.to_dict()
    features = FeatureConfig(alignment_mode=config.audio_alignment_mode,
                             before_samples=config.audio_samples_before,
                             after_samples=config.audio_samples_after)
    previous_features = FeatureConfig(alignment_mode=previous.audio_alignment_mode,
                                      before_samples=previous.audio_samples_before,
                                      after_samples=previous.audio_samples_after)
    adc = np.arange(48000, dtype=np.uint16).reshape(8000, 6)
    sample = {"adc": adc, "audio_before": 4000, "audio_after": 4000}
    np.testing.assert_array_equal(extract_event_audio(sample, config=features),
                                  extract_event_audio(sample, config=previous_features))
    model = build_model(MODEL_NAME, config).eval()
    path = tmp_path / "transformer_contact.pt"
    torch.save({"format_version": 1, "model_name": MODEL_NAME, "model_config": config.to_dict(),
                "feature_config": features.to_dict(), "model": model.state_dict()}, path)
    restored, _, restored_features = load_checkpoint(path)
    restored.eval()
    assert restored_features == features
    image = torch.randn(1, 3, 224, 224)
    audio = torch.randn(1, 6, 64, config.audio_frames)
    contact = torch.tensor([1])
    with torch.inference_mode():
        expected, actual = model(image, audio, contact), restored(image, audio, contact)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

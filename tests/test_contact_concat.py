import pytest
import torch

from mvts.architectures import available_models, build_model
from mvts.engine import load_checkpoint
from mvts.features import FeatureConfig
from mvts.model_contact_concat_sensor_final import ModelConfig
from mvts.model_sensor_final import ModelConfig as BaseModelConfig


MODEL_NAME = "model_contact_concat_sensor_final"


def test_contact_reaches_both_heads_after_attention_and_backpropagates():
    torch.manual_seed(19)
    model = build_model(MODEL_NAME, {"dropout": 0}).eval()
    image = torch.randn(1, 3, 224, 224).repeat(2, 1, 1, 1)
    audio = torch.randn(1, 6, 64, 12).repeat(2, 1, 1, 1)
    inputs = {}

    def capture(name):
        def hook(_module, args):
            inputs[name] = args[0].detach().clone()
        return hook

    hooks = [module.register_forward_pre_hook(capture(name)) for name, module in (
        ("transformer", model.transformer), ("position", model.position_head), ("wrench", model.force_head))]
    output = model(image, audio, torch.tensor([0, 1]))
    for hook in hooks:
        hook.remove()
    assert model.contact_embedding is None
    assert inputs["transformer"].shape == (2, 13, 256)
    torch.testing.assert_close(inputs["transformer"][0], inputs["transformer"][1])
    assert inputs["position"].shape == (2, 257)
    torch.testing.assert_close(inputs["position"], inputs["wrench"])
    torch.testing.assert_close(inputs["position"][0, :-1], inputs["position"][1, :-1])
    torch.testing.assert_close(inputs["position"][:, -1], torch.tensor([0., 1.]))
    assert output["position"].shape == (2, 3)
    assert output["wrench"].shape == (2, 6)
    for value in output.values():
        assert not torch.allclose(value[0], value[1])
    loss = output["position"].square().mean() + output["wrench"].square().mean()
    loss.backward()
    for weight in (model.vision_backbone.stem[0].weight, model.audio_backbone.stem[0].weight,
                   model.position_head[1].weight, model.force_head[1].weight):
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    for head in (model.position_head, model.force_head):
        assert head[1].weight.grad[:, -1].abs().sum() > 0


def test_binary_contact_validation():
    model = build_model(MODEL_NAME).eval()
    image, audio = torch.zeros(2, 3, 224, 224), torch.zeros(2, 6, 64, 12)
    for contact in (None, torch.tensor(1), torch.tensor([0]), torch.tensor([[0, 1]]),
                    torch.tensor([0.5, 1]), torch.tensor([-1, 1]), torch.tensor([0, 2]),
                    torch.tensor([float("nan"), 1]), torch.tensor([0, float("inf")])):
        with pytest.raises(ValueError, match="starts_in_contact"):
            model(image, audio, contact)
    with torch.inference_mode():
        integer = model(image, audio, torch.tensor([0, 1]))
        boolean = model(image, audio, torch.tensor([[False], [True]]))
    for key in integer:
        torch.testing.assert_close(integer[key], boolean[key])
    with pytest.raises(ValueError, match="use_contact"):
        build_model(MODEL_NAME, {"use_contact": False})


def test_model_registration_alignment_and_checkpoint_roundtrip(tmp_path):
    assert MODEL_NAME in available_models()
    config, base = ModelConfig(), BaseModelConfig()
    assert (config.audio_alignment_mode, config.audio_samples_before, config.audio_samples_after) == (
        base.audio_alignment_mode, base.audio_samples_before, base.audio_samples_after)
    features = FeatureConfig(alignment_mode=config.audio_alignment_mode,
                             before_samples=config.audio_samples_before,
                             after_samples=config.audio_samples_after)
    model = build_model(MODEL_NAME, config).eval()
    path = tmp_path / "contact.pt"
    torch.save({"format_version": 1, "model_name": MODEL_NAME, "model_config": config.to_dict(),
                "feature_config": features.to_dict(), "model": model.state_dict()}, path)
    restored, _, restored_features = load_checkpoint(path)
    restored.eval()
    assert restored.config.use_contact is True
    assert restored_features == features
    image, audio, contact = torch.randn(1, 3, 224, 224), torch.randn(1, 6, 64, 12), torch.tensor([1])
    with torch.inference_mode():
        expected, actual = model(image, audio, contact), restored(image, audio, contact)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

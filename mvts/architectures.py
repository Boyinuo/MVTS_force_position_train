"""Discover local model*.py modules and build their shared model interface."""
from importlib import import_module
from pathlib import Path


DEFAULT_MODEL = "model_triple_head"
LEGACY_MODELS = {"model": "model_sensor_nomal",
                 "model_standard_audio": "model_standard_audio_sensor_nomal"}


def available_models():
    names = {path.stem for path in Path(__file__).parent.glob("model*.py")
             if path.is_file() and path.stem.isidentifier()}
    return sorted(names | LEGACY_MODELS.keys())


def get_model_module(name=DEFAULT_MODEL):
    if name not in available_models():
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(available_models())}")
    module = import_module(f".{LEGACY_MODELS.get(name, name)}", package=__package__)
    if not hasattr(module, "ModelConfig") or not hasattr(module, "TactileFusionNet"):
        raise ValueError(f"mvts/{name}.py must expose ModelConfig and TactileFusionNet")
    return module


def build_model(name=DEFAULT_MODEL, config=None):
    module = get_model_module(name)
    if config is None or isinstance(config, dict):
        config = module.ModelConfig(**(config or {}))
    model = module.TactileFusionNet(config)
    model.model_name = name
    return model

"""Discover local model modules/packages and build their shared interface."""
from importlib import import_module
from pathlib import Path


DEFAULT_MODEL = "model_triple_head"
DUAL_RATE_MODELS = frozenset(("model_triple_head", "model_slip_gated"))
LEGACY_MODELS = {"model": "model_sensor_nomal",
                 "model_standard_audio": "model_standard_audio_sensor_nomal"}


def available_models():
    root = Path(__file__).parent
    names = {path.stem for path in root.glob("model*.py")
             if path.is_file() and path.stem.isidentifier()}
    names.update(path.name for path in root.glob("model*")
                 if path.is_dir() and path.name.isidentifier()
                 and (path / "__init__.py").is_file()
                 and (path / f"{path.name}.py").is_file())
    return sorted(names | LEGACY_MODELS.keys())


def get_model_module(name=DEFAULT_MODEL):
    if name not in available_models():
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(available_models())}")
    module = import_module(f".{LEGACY_MODELS.get(name, name)}", package=__package__)
    if not hasattr(module, "ModelConfig") or not hasattr(module, "TactileFusionNet"):
        raise ValueError(f"Model {name!r} must expose ModelConfig and TactileFusionNet")
    return module


def build_model(name=DEFAULT_MODEL, config=None):
    module = get_model_module(name)
    if config is None or isinstance(config, dict):
        config = module.ModelConfig(**(config or {}))
    model = module.TactileFusionNet(config)
    model.model_name = name
    return model

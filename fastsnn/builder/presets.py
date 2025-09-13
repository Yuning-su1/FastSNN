from .config import SNNConfig

_PRESETS = {
    "snn-tiny": dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024),
    "snn-small": dict(d_model=512, n_heads=8, n_layers=8, d_ff=2048),
    "snn-base": dict(d_model=768, n_heads=12, n_layers=12, d_ff=3072),
}

def get_preset(name: str) -> SNNConfig:
    if name not in _PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    cfg = SNNConfig()
    for k, v in _PRESETS[name].items():
        setattr(cfg, k, v)
    return cfg

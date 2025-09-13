# fastsnn/ffn/__init__.py
from .pulse_ffn import PulseFFN, PulseFFNConfig
from .pulse_moe import PulseMoE, PulseMoEConfig, MoEAuxLoss

__all__ = [
    "PulseFFN", "PulseFFNConfig",
    "PulseMoE", "PulseMoEConfig", "MoEAuxLoss",
]

def build_ffn(kind: str, **kwargs):
    """
    Factory to build FFN by kind.
    kind: "pulse" | "pulse_moe"
    """
    kind = (kind or "pulse").lower()
    if kind == "pulse":
        return PulseFFN(**kwargs)
    elif kind in ("moe", "pulse_moe", "pulse-moe"):
        return PulseMoE(**kwargs)
    else:
        raise ValueError(f"Unknown FFN kind: {kind}")

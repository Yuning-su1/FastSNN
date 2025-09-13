from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json, yaml, copy

@dataclass
class SNNConfig:
    # minimal, extensible config used by build_model_from_config
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    rope_base: int = 1_000_000

    # attention section
    attn_kind: str = "linear"          # linear | sliding | hybrid_alt | hybrid_mix
    attn_phi: str = "relu"
    sw_window: int = 128               # sliding-window size (if used)

    # neuron section
    neuron_type: str = "lif_sint"      # lif_sint | adaptive_threshold | adex_sint
    neuron_tau: float = 0.5
    neuron_theta: float = 1.0
    neuron_theta_learnable: bool = True
    neuron_ste_tau: float = 1.0

    # training/inference misc
    dropout_p: float = 0.0
    tie_lm_head: bool = True

def to_nested_dict(cfg: "SNNConfig") -> Dict[str, Any]:
    return {
        "model": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_ff": cfg.d_ff,
            "rope_base": cfg.rope_base,
            "attn_kind": cfg.attn_kind,
            "attn_phi": cfg.attn_phi,
            "sw_window": cfg.sw_window,
            "neuron_type": cfg.neuron_type,
            "neuron_tau": cfg.neuron_tau,
            "neuron_theta": cfg.neuron_theta,
            "neuron_theta_learnable": cfg.neuron_theta_learnable,
            "neuron_ste_tau": cfg.neuron_ste_tau,
            "dropout_p": cfg.dropout_p,
            "tie_lm_head": cfg.tie_lm_head,
        }
    }

def save_config(cfg: "SNNConfig", path: str):
    nested = to_nested_dict(cfg)
    if path.endswith(".json"):
        json.dump(nested, open(path, "w"), indent=2)
    else:
        yaml.safe_dump(nested, open(path, "w"))

def load_config(path: str) -> SNNConfig:
    if path.endswith(".json"):
        d = json.load(open(path))
    else:
        d = yaml.safe_load(open(path))
    m = d.get("model", d)
    return SNNConfig(**m)

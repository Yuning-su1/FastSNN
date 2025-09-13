"""
Adapters to build SNN models from HuggingFace configs or checkpoints.
"""
from typing import Optional
import torch, torch.nn as nn

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

from .config import SNNConfig
from .model_from_config import build_model_from_config
from .ann2snn import convert_ann_to_snn

def build_from_hf(name_or_path: str, cfg: Optional[SNNConfig] = None) -> nn.Module:
    """
    Load an HF causal LM and convert to an SNN skeleton with best-effort reuse.
    """
    if not _HF_AVAILABLE:
        raise RuntimeError("transformers not installed. pip install transformers")

    ann = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.float32)
    if cfg is None:
        # infer config from HF model
        hfc = ann.config
        cfg = SNNConfig(
            vocab_size=getattr(hfc, "vocab_size", 50257),
            d_model=getattr(hfc, "hidden_size", 256),
            n_heads=getattr(hfc, "num_attention_heads", 4),
            n_layers=getattr(hfc, "num_hidden_layers", 4),
            d_ff=getattr(hfc, "intermediate_size", 1024),
        )
    return convert_ann_to_snn(ann, cfg)

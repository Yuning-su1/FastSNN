"""
Utility helpers for context extension and positional settings.
"""
from typing import Optional
import torch, torch.nn as nn
from .model_from_config import SNNLanguageModel

def extend_context_length(model: nn.Module, new_seq_len: int, warmup_ratio: float = 0.2):
    """
    Book-keeping hook for 'conversion-period' context extension:
    - in our minimal model, RoPE params live outside; here we simply tag metadata or
      adjust any cached buffer if you add one later.
    - return a dict that trainer can use to reduce LR (warmup) for N steps.
    """
    # In a real RoPE setup, you would adjust rope_base / interpolation here.
    meta = {
        "new_seq_len": new_seq_len,
        "warmup_ratio": warmup_ratio,
        "note": "Apply small LR warmup on first steps after extension."
    }
    return meta

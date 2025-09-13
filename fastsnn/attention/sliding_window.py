# attention/sliding_window.py
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def _build_band_mask(T: int, window: int, device, dtype) -> torch.Tensor:
    """
    Build an additive attention mask for causal sliding window.
    Allowed: keys in [t-window, t], Disallowed: -inf.
    Returns shape [T, T] to be broadcast to [B*H, T, T].
    """
    # base causal mask: allow i >= j
    idx = torch.arange(T, device=device)
    causal = (idx[:, None] >= idx[None, :])  # [T,T]
    if window is not None and window > 0:
        band = (idx[:, None] - idx[None, :]) <= window
        allowed = causal & band
    else:
        allowed = causal
    mask = torch.zeros(T, T, device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask

class SlidingWindowAttention(nn.Module):
    """
    Causal sliding-window attention using PyTorch SDPA.
    Efficient banded mask construction; avoids Python loops in T.
    Shapes:
      x: [B, T, D]  → out: [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int = 128,
        dropout_p: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_heads)

        # merge batch & head for SDPA
        q = rearrange(q, "b h t d -> (b h) t d", b=B, h=self.n_heads)
        k = rearrange(k, "b h t d -> (b h) t d", b=B, h=self.n_heads)
        v = rearrange(v, "b h t d -> (b h) t d", b=B, h=self.n_heads)

        # additive mask [T,T]
        mask = _build_band_mask(T, self.window, device, q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,       # additive mask; -inf blocks
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,      # causal is already encoded in mask
        )
        out = rearrange(out, "(b h) t d -> b t (h d)", b=B, h=self.n_heads)
        return self.out_proj(out)

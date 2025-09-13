
from __future__ import annotations
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

def _band_mask(T: int, window: int, device, dtype) -> torch.Tensor:
    idx = torch.arange(T, device=device)
    i = idx[:, None]
    j = idx[None, :]
    causal = (i >= j)
    local = (i - j) <= window
    mask = torch.where(causal & local, torch.zeros((), device=device, dtype=dtype), torch.full((), -float('inf'), device=device, dtype=dtype))
    return mask

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int = 128, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = int(window)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = float(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        q = rearrange(self.q_proj(x), 'b t (h d) -> (b h) t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> (b h) t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> (b h) t d', h=self.n_heads)
        mask = _band_mask(T, self.window, x.device, q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)
        out = rearrange(out, '(b h) t d -> b t (h d)', h=self.n_heads, b=B)
        return self.out_proj(out)

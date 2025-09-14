
from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch, torch.nn as nn
from .linear import LinearAttention
from .sliding_window import SlidingWindowAttention
from fastsnn.core.spike_tensor import to_count_if_spike, wrap_like_input

class SoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.dp = float(dropout_p)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        from einops import rearrange
        B,T,D = x.size()
        h = self.n_heads
        q = rearrange(self.q(x), 'b t (h d) -> (b h) t d', h=h)
        k = rearrange(self.k(x), 'b t (h d) -> (b h) t d', h=h)
        v = rearrange(self.v(x), 'b t (h d) -> (b h) t d', h=h)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dp if self.training else 0.0)
        out = rearrange(out, '(b h) t d -> b t (h d)', h=h, b=B)
        return self.out(out)

class HybridMixAttention(nn.Module):
    """Mix Linear + Sliding (+ optional Softmax) and project back to model dim."""
    def __init__(self, d_model: int, n_heads: int, window: int = 128, include_softmax: bool = False, dropout_p: float = 0.0):
        super().__init__()
        self.linear = LinearAttention(d_model, n_heads, dropout_p=dropout_p)
        self.sliding = SlidingWindowAttention(d_model, n_heads, window=window, dropout_p=dropout_p)
        self.softmax = SoftmaxAttention(d_model, n_heads, dropout_p=dropout_p) if include_softmax else None
        self.out_proj = nn.Linear(d_model* (2 + (1 if include_softmax else 0)) , d_model)
    def forward(self, x: torch.Tensor, kv_state: Optional[Dict[str,torch.Tensor]] = None, incremental: bool = False) -> Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        
        out_lin, state = self.linear(x, kv_state=kv_state, incremental=incremental)
        out_swa = self.sliding(x)
        outs = [out_lin, out_swa]
        if self.softmax is not None:
            outs.append(self.softmax(x))
        y = torch.cat(outs, dim=-1)
        return self.out_proj(y), state

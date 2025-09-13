# attention/hybrid.py
from __future__ import annotations
from typing import Literal, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import LinearAttention
from .sliding_window import SlidingWindowAttention

HybridKind = Literal["linear+sliding", "linear+sliding+softmax"]
GateGranularity = Literal["per_layer", "per_head"]

class SoftmaxAttention(nn.Module):
    """Vanilla causal softmax attention for hybrid mixing."""
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale = (self.d_head ** -0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        # causal
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)

class HybridAttention(nn.Module):
    """
    Fuse multiple attention branches:
      - linear (efficient global via kernelized prefix)
      - sliding (local window)
      - softmax (exact, optional)
    Fusion uses a learnable gate (softmax-normalized).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kind: HybridKind = "linear+sliding",
        phi: str = "softplus",
        sw_window: int = 128,
        gate_granularity: GateGranularity = "per_layer",
        dropout_p: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.kind = kind
        self.linear = LinearAttention(d_model, n_heads, phi=phi, dropout_p=dropout_p, bias=bias)
        self.sliding = SlidingWindowAttention(d_model, n_heads, window=sw_window, dropout_p=dropout_p, bias=bias)
        self.use_softmax = (kind == "linear+sliding+softmax")
        if self.use_softmax:
            self.softmax = SoftmaxAttention(d_model, n_heads, dropout_p=dropout_p, bias=bias)

        self.gate_granularity = gate_granularity
        branches = 3 if self.use_softmax else 2

        if gate_granularity == "per_layer":
            # one gate for whole layer (scalar per branch)
            self.gate_logits = nn.Parameter(torch.zeros(branches))
        elif gate_granularity == "per_head":
            # per-head gate: [H, branches]
            self.gate_logits = nn.Parameter(torch.zeros(n_heads, branches))
        else:
            raise ValueError(f"Unknown gate granularity: {gate_granularity}")

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _mix(self, outs: list[torch.Tensor]) -> torch.Tensor:
        """
        outs: list of [B,T,D], length = 2 or 3
        Apply softmax over gate logits and fuse.
        """
        B, T, D = outs[0].shape
        if self.gate_granularity == "per_layer":
            w = torch.softmax(self.gate_logits, dim=-1)  # [K]
            y = sum(w[i] * outs[i] for i in range(len(outs)))
            return y
        else:
            # per-head mixing: reshape to [B,H,T,Dh] and weight per head
            H = self.linear.n_heads
            Dh = D // H
            outs_h = [o.view(B, T, H, Dh).transpose(1, 2) for o in outs]  # [B,H,T,Dh]
            w = torch.softmax(self.gate_logits, dim=-1)  # [H, K]
            # y_h = Σ_k w[h,k] * out_k[h]
            y_h = sum(outs_h[k] * w[:, k].view(1, H, 1, 1) for k in range(len(outs)))
            y = y_h.transpose(1, 2).contiguous().view(B, T, D)
            return y

    def forward(
        self,
        x: torch.Tensor,
        kv_state: Optional[Dict[str, torch.Tensor]] = None,
        incremental: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outs = []
        out_lin, state = self.linear(x, kv_state=kv_state, incremental=incremental)
        outs.append(out_lin)
        out_swa = self.sliding(x)
        outs.append(out_swa)
        if self.use_softmax:
            outs.append(self.softmax(x))
        y = self._mix(outs)
        return self.out_proj(y), state

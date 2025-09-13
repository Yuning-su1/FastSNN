# attention/linear.py
from __future__ import annotations
import math
from typing import Optional, Literal, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

PhiKind = Literal["softplus", "relu", "exp"]

def _phi(x: torch.Tensor, kind: PhiKind = "softplus") -> torch.Tensor:
    """
    Non-negative kernel feature map for linear attention.
    Keeps attention weights ≥ 0 (stability requirement for linearization).
    """
    if kind == "softplus":
        return F.softplus(x) + 1e-6
    elif kind == "relu":
        return F.relu(x) + 1e-6
    elif kind == "exp":
        # precise softmax-kernel; can be less stable on fp16
        return torch.exp(x)
    else:
        raise ValueError(f"Unknown phi kind: {kind}")

class LinearAttention(nn.Module):
    """
    Causal linear attention with non-negative kernel features.
    Forward supports two modes:
      - full: iterate time dimension with prefix states (stable & streaming-friendly)
      - incremental: pass kv_state to update running prefix (for fast autoregressive decoding)
    Shapes:
      x: [B, T, D]  → out: [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        phi: PhiKind = "softplus",
        dropout_p: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.phi_kind = phi
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    @torch.no_grad()
    def _init_state(self, B: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # kv_acc: [B, H, D_head, D_head] ; z_acc: [B, H, D_head]
        kv_acc = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=device, dtype=dtype)
        z_acc = torch.zeros(B, self.n_heads, self.d_head, device=device, dtype=dtype)
        return kv_acc, z_acc

    def forward(
        self,
        x: torch.Tensor,
        kv_state: Optional[Dict[str, torch.Tensor]] = None,
        incremental: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
          x: [B, T, D]
          kv_state: running prefix states for incremental decoding
          incremental: if True, treat x as a continuation (T small, e.g., 1)
        Returns:
          out: [B, T, D]
          new_state: dict with kv_acc, z_acc for next step (if incremental or requested)
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        qf = _phi(q, self.phi_kind)               # [B,H,T,Dh]
        kf = _phi(k, self.phi_kind)               # [B,H,T,Dh]

        if incremental:
            # Streaming: update prefix with the provided chunk (usually T=1)
            if kv_state is None:
                kv_acc, z_acc = self._init_state(B, device, dtype)
            else:
                kv_acc, z_acc = kv_state["kv_acc"], kv_state["z_acc"]

            outs = []
            one_vec = torch.ones(B, self.n_heads, self.d_head, 1, device=device, dtype=dtype)  # for z updates
            for t in range(T):
                kt = kf[:, :, t : t + 1, :]                       # [B,H,1,Dh]
                vt = v[:, :, t : t + 1, :]                        # [B,H,1,Dh]
                # Update prefix states
                kv_acc = kv_acc + torch.einsum("b h 1 d, b h 1 e -> b h d e", kt, vt)
                z_acc = z_acc + kt.squeeze(2)                     # sum of k-features

                qt = qf[:, :, t : t + 1, :]                       # [B,H,1,Dh]
                num = torch.einsum("b h 1 d, b h d e -> b h 1 e", qt, kv_acc)   # [B,H,1,Dh]
                den = torch.einsum("b h 1 d, b h d -> b h 1", qt, z_acc) + 1e-6  # [B,H,1]
                ot = num / den.unsqueeze(-1)
                outs.append(ot)
            out = torch.cat(outs, dim=2)                          # [B,H,T,Dh]
            new_state = {"kv_acc": kv_acc, "z_acc": z_acc}

        else:
            # Full pass: causal by prefix scanning (loop over T for stability & clarity)
            kv_acc, z_acc = self._init_state(B, device, dtype)
            outs = []
            for t in range(T):
                kt = kf[:, :, t : t + 1, :]
                vt = v[:, :, t : t + 1, :]
                kv_acc = kv_acc + torch.einsum("b h 1 d, b h 1 e -> b h d e", kt, vt)
                z_acc = z_acc + kt.squeeze(2)
                qt = qf[:, :, t : t + 1, :]
                num = torch.einsum("b h 1 d, b h d e -> b h 1 e", qt, kv_acc)
                den = torch.einsum("b h 1 d, b h d -> b h 1", qt, z_acc) + 1e-6
                ot = num / den.unsqueeze(-1)
                outs.append(ot)
            out = torch.cat(outs, dim=2)
            new_state = {"kv_acc": kv_acc, "z_acc": z_acc}

        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(self.dropout(out))
        return out, new_state

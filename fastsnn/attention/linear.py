
from __future__ import annotations
from typing import Optional, Literal, Dict, Tuple
import torch, torch.nn as nn
from einops import rearrange

PhiKind = Literal['softplus','relu','exp']

def _phi(x: torch.Tensor, kind: PhiKind = 'softplus') -> torch.Tensor:
    if kind == 'softplus':
        return torch.nn.functional.softplus(x)
    if kind == 'relu':
        return torch.relu(x)
    if kind == 'exp':
        return torch.exp(x)
    raise ValueError(kind)

class LinearAttention(nn.Module):
    """Gated Linear Attention (state-space style) with O(T) complexity.
    Returns (out, state) where state holds accumulators for incremental decoding.
    """
    def __init__(self, d_model: int, n_heads: int, phi: PhiKind = 'softplus', dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gate = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout_p)
        self.phi = phi

    def forward(self, x: torch.Tensor, kv_state: Optional[Dict[str,torch.Tensor]] = None, incremental: bool = False) -> Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        B, T, D = x.size()
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        g = rearrange(torch.sigmoid(self.gate(x)), 'b t (h d) -> b h t d', h=self.n_heads)

        qf = _phi(q, self.phi) * (g + 1e-6)  # gating ensures stability
        kf = _phi(k, self.phi)

        if incremental and kv_state is not None:
            kv_acc = kv_state['kv_acc']  # [B,H,D,D]
            z_acc  = kv_state['z_acc']   # [B,H,D]
        else:
            kv_acc = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device, dtype=x.dtype)
            z_acc  = torch.zeros(B, self.n_heads, self.d_head, device=x.device, dtype=x.dtype)

        if incremental:
            kt = kf[:, :, -1:, :]  # [B,H,1,D]
            vt = v[:, :, -1:, :]
            kv_acc = kv_acc + torch.einsum('b h 1 d, b h 1 e -> b h d e', kt, vt)
            z_acc  = z_acc  + kt.squeeze(2)
            qt = qf[:, :, -1:, :]
            num = torch.einsum('b h 1 d, b h d e -> b h 1 e', qt, kv_acc)
            den = torch.einsum('b h 1 d, b h d -> b h 1', qt, z_acc) + 1e-6
            out = num / den.unsqueeze(-1)
            new_state = {'kv_acc': kv_acc, 'z_acc': z_acc}
        else:
            outs = []
            for t in range(T):
                kt = kf[:, :, t:t+1, :]
                vt = v[:, :, t:t+1, :]
                kv_acc = kv_acc + torch.einsum('b h 1 d, b h 1 e -> b h d e', kt, vt)
                z_acc  = z_acc  + kt.squeeze(2)
                qt = qf[:, :, t:t+1, :]
                num = torch.einsum('b h 1 d, b h d e -> b h 1 e', qt, kv_acc)
                den = torch.einsum('b h 1 d, b h d -> b h 1', qt, z_acc) + 1e-6
                outs.append(num / den.unsqueeze(-1))
            out = torch.cat(outs, dim=2)
            new_state = {'kv_acc': kv_acc, 'z_acc': z_acc}

        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(self.dropout(out)), new_state

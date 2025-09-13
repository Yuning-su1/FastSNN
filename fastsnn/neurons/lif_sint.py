
from __future__ import annotations
import torch, torch.nn as nn
from .surrogate import SurrogateSTE

class LIFsINT(nn.Module):
    """Simplified LIF neuron that outputs spike-count integers (sINT).
    Collapses time during training; soft-reset.
    Input: x [B,T,D]; Output: sINT [B,T,D] (broadcasted count per timestep)
    """
    def __init__(self, d_model: int, theta: float = 1.0, learn_theta: bool = True, ste_tau: float = 1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.full((d_model,), float(theta)), requires_grad=learn_theta)
        self.ste_tau = float(ste_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sum over time -> v; divide by adaptive threshold (per-dim)
        v = x.sum(dim=1, keepdim=True)  # [B,1,D]
        v_over_theta = v / self.theta.view(1,1,-1)
        n = SurrogateSTE.apply(v_over_theta, self.ste_tau)  # [B,1,D] integers
        return n.expand_as(x)

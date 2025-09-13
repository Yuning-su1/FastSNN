
from __future__ import annotations
import torch, torch.nn as nn
from .surrogate import SurrogateSTE

class AdaptiveThresholdNeuron(nn.Module):
    """Adaptive-threshold spike-count neuron (IF with soft reset).
    Threshold depends on mean absolute membrane potential: Vth = mean(|x|)/k.
    This follows the simplified scheme in SpikingBrain for sINT generation.
    Input: x [B,T,D] ; Output: sINT [B,T,D]
    """
    def __init__(self, d_model: int, k: float = 3.0, ste_tau: float = 1.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k)), requires_grad=False)
        self.ste_tau = float(ste_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # collapse time
        v = x.sum(dim=1, keepdim=True)  # [B,1,D]
        # dynamic threshold from batch statistics
        Vth = (x.abs().mean(dim=(1,)) / self.k).unsqueeze(1)  # [B,1,D] via broadcast
        Vth = torch.clamp(Vth, min=1e-5)
        v_over = v / Vth
        n = SurrogateSTE.apply(v_over, self.ste_tau)
        return n.expand_as(x)

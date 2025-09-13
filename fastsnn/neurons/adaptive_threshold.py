# fastsnn/neurons/adaptive_threshold.py
import torch, torch.nn as nn
from .surrogate import SurrogateSTE

class AdaptiveThresholdNeuron(nn.Module):
    """
    Adaptive-threshold IF neuron with single-step sINT conversion (training) and soft-reset.
    Implements the scheme in SpikingBrain (Eq. 22–25):
        Vth(x) = mean(abs(x)) / k
        sINT   = round( x / Vth(x) )
    During training we collapse the temporal dimension and return an integer-like activation
    via STE; for inference one can expand sINT into a sparse spike train.
    """
    def __init__(self, d_model, k: float = 5.0, ste_tau: float = 1.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k)), requires_grad=False)
        self.ste_tau = ste_tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D] continuous activations (projection output)
        v = x.sum(dim=1, keepdim=True)              # collapse time during training
        vth = (v.abs().mean(dim=(1,2), keepdim=True) / (self.k + 1e-6))  # [B,1,1]
        v_over = v / (vth + 1e-6)
        n_int = SurrogateSTE.apply(v_over, self.ste_tau)
        return n_int.expand_as(x)

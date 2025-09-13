# fastsnn/neurons/adex_sint.py
import torch, torch.nn as nn
from .surrogate import SurrogateSTE

class AdExsINT(nn.Module):
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) 简化计数版。
    - 加入 w (适应电流) 抑制过度放电
    """
    def __init__(self, d_model, tau=0.5, theta=1.0, beta=0.05, learn_theta=True, ste_tau=1.0):
        super().__init__()
        self.alpha = tau
        self.theta = nn.Parameter(torch.full((d_model,), theta), requires_grad=learn_theta)
        self.beta = nn.Parameter(torch.full((d_model,), beta), requires_grad=True)
        self.ste_tau = ste_tau

    def forward(self, x):
        v = x.sum(dim=1, keepdim=True)       # 膜电位累计
        adaptive = self.beta.view(1,1,-1) * v
        v_eff = v - adaptive                 # 适应抑制
        v_over_theta = v_eff / self.theta.view(1,1,-1)
        n_int = SurrogateSTE.apply(v_over_theta, self.ste_tau)
        return n_int.expand_as(x)

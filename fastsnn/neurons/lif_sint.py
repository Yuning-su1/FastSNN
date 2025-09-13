# fastsnn/neurons/lif_sint.py
import torch, torch.nn as nn
from .surrogate import SurrogateSTE

class LIFsINT(nn.Module):
    """
    Leak-Integrate-and-Fire with spike-count integer (sINT).
    - 输入 [B,T,D]，输出整数脉冲计数 [B,T,D]
    - soft-reset + 合并时间维度
    """
    def __init__(self, d_model, tau=0.5, theta=1.0, learn_theta=True, ste_tau=1.0):
        super().__init__()
        self.alpha = tau
        self.theta = nn.Parameter(torch.full((d_model,), theta), requires_grad=learn_theta)
        self.ste_tau = ste_tau

    def forward(self, x):
        # x: [B,T,D]
        v = x.sum(dim=1, keepdim=True)   # 简化：合并时间
        v_over_theta = v / self.theta.view(1,1,-1)
        n_int = SurrogateSTE.apply(v_over_theta, self.ste_tau)
        return n_int.expand_as(x)

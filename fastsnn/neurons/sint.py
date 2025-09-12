import torch, torch.nn as nn, torch.nn.functional as F

class STECount(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_over_theta, ste_tau: float):
        # v_over_theta: 任意实数，代表按阈值归一化后的“发放强度”
        ctx.save_for_backward(v_over_theta)
        ctx.ste_tau = ste_tau
        return torch.clamp(v_over_theta.round(), 0, None)  # 非负整数计数（可接近 0/1/2...）
    @staticmethod
    def backward(ctx, grad_out):
        (v_over_theta,) = ctx.saved_tensors
        tau = ctx.ste_tau
        # 替代梯度：σ'(x/τ) ~ (σ(x/τ)*(1-σ(x/τ)))/τ
        sig = torch.sigmoid(v_over_theta / tau)
        grad_in = grad_out * sig * (1 - sig) / max(tau, 1e-6)
        return grad_in, None

class LIFsINT(nn.Module):
    def __init__(self, d_model, tau=0.5, theta_init=1.0, theta_learnable=True, ste_tau=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(tau), requires_grad=False) # 可改成固定
        self.theta = nn.Parameter(torch.full((d_model,), theta_init), requires_grad=theta_learnable)
        self.ste_tau = ste_tau
        self.register_buffer("v", torch.zeros(1, 1, d_model))  # 简化：每次重置或按需扩展

    def forward(self, x):
        # x: [B, T, d], 连续输入（上一算子输出）
        # 累积膜电位（同步窗口内一次性处理）
        # 简化：把时间维合并成一次累加（测试版）
        # v_acc = alpha * v_prev + sum_t x_t
        B, T, D = x.shape
        v_prev = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        v_acc = self.alpha * v_prev + x.sum(dim=1, keepdim=True)  # [B,1,D]
        v_over_theta = v_acc / (self.theta.view(1, 1, -1))
        n_int = STECount.apply(v_over_theta, self.ste_tau)        # [B,1,D] 整数/近似整数激活
        return n_int.expand(B, T, D)  # 回填到 T 位置，保持形状兼容（测试版简化）

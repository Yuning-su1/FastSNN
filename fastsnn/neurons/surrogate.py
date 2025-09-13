# fastsnn/neurons/surrogate.py
import torch

class SurrogateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_over_theta, ste_tau: float):
        ctx.save_for_backward(v_over_theta)
        ctx.ste_tau = ste_tau
        return torch.clamp(v_over_theta.round(), 0, None)  # spike count (整数)

    @staticmethod
    def backward(ctx, grad_out):
        (v_over_theta,) = ctx.saved_tensors
        tau = ctx.ste_tau
        sig = torch.sigmoid(v_over_theta / tau)
        grad_in = grad_out * sig * (1 - sig) / max(tau, 1e-6)
        return grad_in, None

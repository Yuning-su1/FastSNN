
import torch

class SurrogateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_over_theta: torch.Tensor, ste_tau: float):
        ctx.save_for_backward(v_over_theta)
        ctx.tau = float(ste_tau)
        # clamp min 0 for spike-count semantics
        return torch.clamp(v_over_theta.round(), 0, None)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (v_over_theta,) = ctx.saved_tensors
        tau = max(ctx.tau, 1e-6)
        sig = torch.sigmoid(v_over_theta / tau)
        grad = grad_out * sig * (1 - sig) / tau
        return grad, None

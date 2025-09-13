# fastsnn/ffn/pulse_moe.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastsnn.ffn.pulse_ffn import PulseFFN
from fastsnn.neurons import build_neuron

@dataclass
class PulseMoEConfig:
    d_model: int
    d_ff: int
    num_experts: int = 4
    router_mode: str = "top1"          # 目前实现 top-1
    capacity_factor: float = 1.25      # token 容量因子（Switch Transformer 风格）
    router_noise: float = 0.0          # 训练时高斯噪声
    aux_loss_weight: float = 1e-2      # 负载均衡损失系数
    dropout: float = 0.0
    bias: bool = False
    neuron_tau: float = 0.5
    neuron_theta_init: float = 1.0
    neuron_theta_learnable: bool = True
    neuron_ste_tau: float = 1.0
    pre_norm: bool = False
    post_norm: bool = False

class MoEAuxLoss(nn.Module):
    """
    负载均衡损失 (Switch Transformer 样式)：
    encourage all experts to be used with similar probability & load.
    """
    def forward(self, router_probs: torch.Tensor, dispatch_mask: torch.Tensor) -> torch.Tensor:
        """
        router_probs: [N, E] softmax 后的概率
        dispatch_mask: [N, E] one-hot top-1 分配
        """
        # importance: sum of router probability per expert
        imp = router_probs.sum(0)                    # [E]
        # load: how many tokens actually dispatched per expert
        load = dispatch_mask.sum(0)                  # [E]
        imp = imp / (imp.sum() + 1e-6)
        load = load / (load.sum() + 1e-6)
        loss = (imp * load).sum() * router_probs.size(1)  # 越接近均匀越大 → 用 (E * Σ imp*load) 的负号
        # 为了让“越均匀越小”，我们使用 1 - loss 的形式
        return 1.0 - loss

class PulseMoE(nn.Module):
    """
    MoE 版脉冲化 FFN：Top-1 路由，每个专家是一个 PulseFFN。
    - 设计目标：进一步稀疏化 FFN 计算，只激活少量专家。
    - 训练时提供负载均衡辅助损失；推理时与普通 FFN 接口一致。
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        router_mode: str = "top1",
        capacity_factor: float = 1.25,
        router_noise: float = 0.0,
        aux_loss_weight: float = 1e-2,
        dropout: float = 0.0,
        bias: bool = False,
        neuron_tau: float = 0.5,
        neuron_theta_init: float = 1.0,
        neuron_theta_learnable: bool = True,
        neuron_ste_tau: float = 1.0,
        pre_norm: bool = False,
        post_norm: bool = False,
    ):
        super().__init__()
        assert router_mode.lower() in ("top1",), "Only top-1 is supported for now."
        self.d_model, self.d_ff = d_model, d_ff
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router_noise = router_noise
        self.aux_loss_weight = aux_loss_weight

        self.pre_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.post_norm = nn.LayerNorm(d_model) if post_norm else nn.Identity()
        self.router = nn.Linear(d_model, num_experts, bias=True)

        # 构造专家列表（共享结构，少引入新参数的前提下，各专家独立权重）
        experts = []
        for _ in range(num_experts):
            experts.append(
                PulseFFN(
                    d_model=d_model, d_ff=d_ff,
                    tau=neuron_tau, theta_init=neuron_theta_init,
                    theta_learnable=neuron_theta_learnable,
                    ste_tau=neuron_ste_tau,
                    dropout=dropout, bias=bias,
                    pre_norm=False, post_norm=False,
                )
            )
        self.experts = nn.ModuleList(experts)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.aux = MoEAuxLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, T, D]
        return: (y, aux_loss)
          y: [B, T, D]
          aux_loss: scalar tensor (or None in eval)
        """
        B, T, D = x.shape
        h = self.pre_norm(x)
        N = B * T
        h_flat = h.reshape(N, D)              # [N, D]

        # Router logits & probs
        logits = self.router(h_flat)          # [N, E]
        if self.training and self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = F.softmax(logits, dim=-1)     # [N, E]

        # Top-1 assignment
        top1 = probs.argmax(dim=-1)           # [N]
        dispatch = F.one_hot(top1, num_classes=self.num_experts).float()  # [N, E]

        # Capacity (per expert)
        # 容量 = capacity_factor * (N / E)
        cap = int(self.capacity_factor * (N / self.num_experts) + 1e-6)

        # 为每个专家收集 token 索引（带容量裁剪）
        ys = torch.zeros_like(h_flat)         # 最终聚合输出
        load_mask_all = []
        start = torch.zeros(self.num_experts, dtype=torch.long, device=h.device)
        # 给每个专家桶分配位置
        # token → expert 的“队列号”
        ranks = torch.cumsum(dispatch, dim=0)  # [N,E] 每个 expert 的累计计数
        # 实际 mask：仅保留队列号<=cap 的 token
        load_mask = (ranks <= cap).float() * dispatch  # [N,E]
        load_mask_all.append(load_mask)

        # 将 token 分发到各专家并返回
        for e in range(self.num_experts):
            mask_e = load_mask[:, e]          # [N]
            if mask_e.sum() == 0:
                continue
            idx = mask_e.nonzero(as_tuple=True)[0]  # token idx
            x_e = h_flat.index_select(0, idx).unsqueeze(0).unsqueeze(0)  # [1,1,n_e,D] → 伪装 [B,T,D]
            y_e = self.experts[e](x_e).squeeze(0).squeeze(0)             # [n_e,D]
            # scatter 回原位置
            ys.index_copy_(0, idx, y_e)

        y = ys.reshape(B, T, D)
        y = self.dropout(y)
        y = self.post_norm(y)

        aux_loss = None
        if self.training and self.aux_loss_weight > 0:
            aux_raw = self.aux(probs.detach(), dispatch)  # 不反传到 router 概率统计
            aux_loss = aux_raw * self.aux_loss_weight

        return y, aux_loss

# fastsnn/ffn/pulse_ffn.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# 依赖你项目里的 sINT 神经元实现
from fastsnn.neurons import build_neuron

@dataclass
class PulseFFNConfig:
    d_model: int
    d_ff: int
    tau: float = 0.5               # 膜电位衰减
    theta_init: float = 1.0        # 初始阈值
    theta_learnable: bool = True
    ste_tau: float = 1.0           # 替代梯度温度
    dropout: float = 0.0
    bias: bool = False
    pre_norm: bool = False         # 若外部已做 LN，可关
    post_norm: bool = False

class PulseFFN(nn.Module):
    """
    脉冲化 FFN： Linear(d_model->d_ff) → LIFsINT(d_ff) → Linear(d_ff->d_model)
    - 用 sINT 作为激活（整数计数 / 连续逼近），保持 ANN 接口。
    - 设计为“少引入新参数”：只有神经元阈值/衰减。
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        tau: float = 0.5,
        theta_init: float = 1.0,
        theta_learnable: bool = True,
        ste_tau: float = 1.0,
        dropout: float = 0.0,
        bias: bool = False,
        pre_norm: bool = False,
        post_norm: bool = False,
    ):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff
        self.pre_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.post_norm = nn.LayerNorm(d_model) if post_norm else nn.Identity()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.neuron = build_neuron(neuron_type, d_model,
                                    tau=tau, theta=theta,
                                    learn_theta=theta_learnable,
                                    ste_tau=ste_tau)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

        # 小初始化，避免 sINT 初期爆激活
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        h = self.pre_norm(x)
        h = self.fc1(h)           # [B,T,d_ff]
        h = self.neuron(h)        # sINT 激活（[B,T,d_ff] 近似非负整数）
        h = self.dropout(h)
        h = self.fc2(h)           # [B,T,D]
        h = self.post_norm(h)
        return h

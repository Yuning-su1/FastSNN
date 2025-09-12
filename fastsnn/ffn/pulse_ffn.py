import torch, torch.nn as nn
from fastsnn.neurons.sint import LIFsINT

class PulseFFN(nn.Module):
    def __init__(self, d_model, d_ff, neuron_cfg):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.neuron = LIFsINT(d_ff, tau=neuron_cfg["tau"],
                              theta_init=neuron_cfg["theta_init"],
                              theta_learnable=neuron_cfg["theta_learnable"],
                              ste_tau=neuron_cfg["ste_tau"])
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        h = self.fc1(x)                # [B,T,d_ff]
        n = self.neuron(h)             # sINT 激活（[B,T,d_ff] 整数/近似整数）
        return self.fc2(n)

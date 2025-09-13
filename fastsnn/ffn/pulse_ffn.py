
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from fastsnn.neurons.lif_sint import LIFsINT

class PulseFFN(nn.Module):
    """A simple FFN that passes pre-activation through an sINT neuron and projects back."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_adaptive: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.neuron = LIFsINT(d_ff)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        s = self.neuron(h)   # spike-count proxy
        h = h + s            # residual inject spikes
        h = self.drop(h)
        return self.fc2(h)


from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from fastsnn.neurons.lif_sint import LIFsINT
from fastsnn.neurons.adaptive_threshold import AdaptiveThresholdNeuron
from fastsnn.core.spike_tensor import to_count_if_spike, wrap_like_input

class PulseFFN(nn.Module):
    """A simple FFN that passes pre-activation through an sINT neuron and projects back."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_adaptive: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.neuron = AdaptiveThresholdNeuron(d_ff)if use_adaptive==True else LIFsINT(d_ff)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        x_in = x
        x = to_count_if_spike(x)  
        y = self.fc2(self.neuron(self.fc1(x)))
        y = wrap_like_input(y, x_in, kind="count")
        return y


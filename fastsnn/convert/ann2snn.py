from __future__ import annotations
import torch, torch.nn as nn
from typing import Callable, Tuple, Dict, Any
from fastsnn.neurons.adaptive_threshold import AdaptiveThresholdNeuron

class TemporalWrapper(nn.Module):
    """Wrap a static layer to accept [B,T,D] by flattening time and restoring."""
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D] or [B,D]
        if x.dim() == 2:
            return self.layer(x)
        B,T,D = x.size()
        y = self.layer(x.view(B*T, D)).view(B,T,-1)
        return y

def convert_ann_to_snn(module: nn.Module, d_model_fallback: int = 128) -> nn.Module:
    """A minimal, safe converter:
    - nn.ReLU/GELU/SiLU -> AdaptiveThresholdNeuron over time
    - nn.Linear preserved (wrapped to handle [B,T,D])
    - nn.Sequential recursively converted
    - Unknown modules are kept as-is
    """
    if isinstance(module, nn.Sequential):
        return nn.Sequential(*[convert_ann_to_snn(m, d_model_fallback) for m in module])
    if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
        D = getattr(module, 'in_features', None) or d_model_fallback
        return AdaptiveThresholdNeuron(D)
    if isinstance(module, nn.Linear):
        return TemporalWrapper(nn.Linear(module.in_features, module.out_features, bias=module.bias is not None))
    # wrap modules that accept [B,D] to [B,T,D]
    if hasattr(module, 'forward'):
        return module
    return module

class TinyANN(nn.Module):
    """Toy ANN (MLP) to demonstrate conversion."""
    def __init__(self, in_dim=128, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def demo_convert_and_run(B=2, T=8, D=128, C=10) -> Tuple[torch.Tensor, torch.Tensor]:
    ann = TinyANN(D, 256, C)
    snn = convert_ann_to_snn(ann, d_model_fallback=D)
    x = torch.randn(B,T,D)
    y_ann = ann(x.view(B*T, D)).view(B,T,C)
    y_snn = snn(x)
    return y_ann, y_snn

if __name__ == '__main__':
    ya, ys = demo_convert_and_run()
    print('OK convert', ya.shape, ys.shape)

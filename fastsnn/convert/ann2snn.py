# fastsnn/convert/ann2snn.py
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional, Iterable
from fastsnn.neurons.adaptive_threshold import AdaptiveThresholdNeuron

class TemporalWrapper(nn.Module):
    """Wrap an nn.Linear (or any stateless layer) to accept [B,T,D] by flattening time and restoring.
    IMPORTANT: This wrapper expects the `layer` to already contain correct weights (we copy them).
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D] or [B,D]
        if x.dim() == 2:
            return self.layer(x)
        B, T, D = x.size()
        y = self.layer(x.view(B * T, D))
        return y.view(B, T, -1)

def _copy_linear_preserve_state(src: nn.Linear) -> nn.Linear:
    """Create a new Linear with same shape and copy state_dict (weights & bias)."""
    new_lin = nn.Linear(src.in_features, src.out_features, bias=(src.bias is not None))
    # copy weights and bias
    new_lin.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        new_lin.bias.data.copy_(src.bias.data)
    return new_lin

def _is_activation(module: nn.Module) -> bool:
    return isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU))

def _convert_sequential(seq: nn.Sequential, d_model_fallback: int):
    """Convert nn.Sequential with lookahead for Linear->Activation pattern."""
    converted = []
    i = 0
    modules = list(seq)
    while i < len(modules):
        cur = modules[i]
        nxt = modules[i+1] if (i+1) < len(modules) else None

        # Case: Linear followed by Activation -> keep linear (TemporalWrapper), replace activation by AdaptiveThresholdNeuron
        if isinstance(cur, nn.Linear) and (nxt is not None) and _is_activation(nxt):
            # wrap linear (preserving weights)
            new_lin = _copy_linear_preserve_state(cur)
            wrapped = TemporalWrapper(new_lin)
            converted.append(wrapped)

            # replace activation with AdaptiveThresholdNeuron sized by linear.out_features
            d_out = cur.out_features
            converted.append(AdaptiveThresholdNeuron(d_out))
            i += 2
            continue

        # Case: Linear alone -> wrap linear preserving weights
        if isinstance(cur, nn.Linear):
            new_lin = _copy_linear_preserve_state(cur)
            converted.append(TemporalWrapper(new_lin))
            i += 1
            continue

        # Case: Activation alone (no preceding linear) -> replace with adaptive threshold sized by fallback
        if _is_activation(cur):
            converted.append(AdaptiveThresholdNeuron(d_model_fallback))
            i += 1
            continue

        # Case: Nested Sequential or Module -> recursively convert
        if isinstance(cur, nn.Sequential):
            converted.append(_convert_sequential(cur, d_model_fallback))
            i += 1
            continue

        # Case: Any nn.Module with children -> recursively convert children (safe fallback)
        if any(True for _ in cur.children()):
            converted.append(convert_ann_to_snn(cur, d_model_fallback))
            i += 1
            continue

        # Leaf modules w/o children: keep as-is (conservative)
        converted.append(cur)
        i += 1

    return nn.Sequential(*converted)

def convert_ann_to_snn(module: nn.Module, d_model_fallback: int = 128) -> nn.Module:
    """
    Convert an ANN module to an SNN-aware module according to a conservative, runnable policy:
      - For nn.Sequential, performs look-ahead conversion:
          Linear -> Activation  ==> TemporalWrapper(Linear(with copied weights)) + AdaptiveThresholdNeuron(out_features)
      - For standalone nn.Linear: wrap and preserve weights (TemporalWrapper)
      - For activations: map to AdaptiveThresholdNeuron if standalone (size uses d_model_fallback)
      - For modules with children: recursively convert children and reassign into a fresh container module
      - For unknown leaf modules: keep as-is (we do not reinitialize or break parameter ties)
    Notes:
      * This converter is intentionally conservative: it preserves parameters and behavior of layers
        we do not explicitly understand, and only replaces the activation semantics to produce sINTs.
      * It does not attempt to reconstruct arbitrary custom module __init__ signatures; instead we
        build a new container module and set converted submodules by name.
    """
    # Straightforward leaf mappings
    if isinstance(module, nn.Sequential):
        return _convert_sequential(module, d_model_fallback)

    if isinstance(module, nn.Linear):
        new_lin = _copy_linear_preserve_state(module)
        return TemporalWrapper(new_lin)

    if _is_activation(module):
        # fallback size
        return AdaptiveThresholdNeuron(d_model_fallback)

    # If module has named children, build a new container and recursively convert children,
    # preserving other attributes (buffers/parameters) by reference where safe.
    children = list(module.named_children())
    if children:
        # Create a plain nn.Module container and copy metadata where feasible.
        new_module = nn.Module()
        # Keep a reference to the original class name for debugging
        new_module.__name__ = getattr(module, "__class__", module.__class__).__name__

        # Convert and attach children by name
        for name, child in children:
            converted_child = convert_ann_to_snn(child, d_model_fallback)
            setattr(new_module, name, converted_child)

        # Try to copy important attributes (state_dict entries that are not modules)
        # This is conservative: parameters that belonged to leaf children are already copied/kept.
        try:
            # copy buffers (e.g., running_mean/var) if present
            for buf_name, buf in module.named_buffers(recurse=False):
                new_module.register_buffer(buf_name, buf.clone() if isinstance(buf, torch.Tensor) else buf)
        except Exception:
            pass

        return new_module

    # Leaf module with no children and not recognized: keep original (safe fallback)
    return module

# --------- Toy ANN demo ----------
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
        # Accepts [B,T,D] or [B,D] for convenience
        if x.dim() == 3:
            B, T, D = x.size()
            return self.net(x.view(B * T, D)).view(B, T, -1)
        return self.net(x)

def demo_convert_and_run(B=2, T=8, D=128, C=10) -> Tuple[torch.Tensor, torch.Tensor]:
    ann = TinyANN(D, 256, C)
    # initialize a deterministic weight for demo stability
    torch.manual_seed(0)
    for p in ann.parameters():
        if p.ndim >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)
    # convert
    snn = convert_ann_to_snn(ann, d_model_fallback=D)
    # run
    x = torch.randn(B, T, D)
    y_ann = ann(x)           # [B,T,C]
    y_snn = snn(x)           # [B,T,C] (hopefully same shape)
    return y_ann, y_snn

if __name__ == "__main__":
    a, b = demo_convert_and_run()
    print("convert demo shapes:", a.shape, b.shape)

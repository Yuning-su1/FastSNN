import torch
from typing import Dict, List, Any

class TensorTap:
    """Simple forward-hook manager to capture tensors from modules."""
    def __init__(self):
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: Dict[str, Any] = {}

    def tap(self, module, name: str, reduce="none"):
        """Attach forward hook to a module. reduce: 'none'|'mean'|'sparsity'."""
        def _hook(_m, _in, out):
            with torch.no_grad():
                t = out.detach()
                if reduce == "mean":
                    val = t.float().mean().item()
                elif reduce == "sparsity":
                    val = (t == 0).float().mean().item()
                else:
                    val = t.cpu()
                self.cache.setdefault(name, []).append(val)
        h = module.register_forward_hook(_hook)
        self.handles.append(h)

    def clear(self):
        self.cache.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

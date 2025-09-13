from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import torch

class SpikeTensor:
    """Unified representation of spiking activity.
    mode: 'dense' (0/1), 'count' (sINT), or 'event' (list of (b,t,d)).
    """
    def __init__(self, data, mode: str = 'dense', meta: Optional[Dict[str,Any]] = None):
        self.data = data
        self.mode = mode
        self.meta = meta or {}

    def to_dense(self, shape: Optional[Tuple[int,...]] = None) -> torch.Tensor:
        if self.mode == 'dense':
            if shape is None:
                return self.data
            return self.data.view(*shape)
        if self.mode == 'count':
            # Expand spike-count across time if target shape is provided
            if shape is None:
                return self.data
            assert len(shape) == 3, "shape should be (B,T,D)"
            B,T,D = shape
            return self.data.expand(B, T, D)
        if self.mode == 'event':
            assert shape is not None, "shape is required for event->dense"
            out = torch.zeros(shape, dtype=torch.float32, device=self.data.device if hasattr(self.data,'device') else 'cpu')
            for (b,t,d) in self.data:
                out[b,t,d] = 1.0
            return out
        raise ValueError(f'Unknown mode {self.mode}')

    def to_count(self) -> torch.Tensor:
        if self.mode == 'count':
            return self.data
        if self.mode == 'dense':
            # collapse time dimension if present (B,T,D)->(B,1,D)
            if self.data.dim() == 3:
                return self.data.sum(dim=1, keepdim=True)
            return self.data
        if self.mode == 'event':
            raise NotImplementedError('event->count not implemented')
        raise ValueError(f'Unknown mode {self.mode}')

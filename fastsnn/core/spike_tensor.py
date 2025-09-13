import torch

class SpikeTensor:
    """
    Unified representation of spiking activity.

    Supports three modes:
    - "dense": 0/1 tensor (bool or float) [B,T,D]
    - "count": integer counts (sINT) [B,T,D]
    - "event": sparse event list (t,i,j) with spike times
    """
    def __init__(self, data, mode="dense", meta=None):
        self.data = data          # torch.Tensor or list of events
        self.mode = mode          # "dense" | "count" | "event"
        self.meta = meta or {}    # e.g. {"dt":1.0, "threshold":1.0}

    def to_dense(self, shape=None):
        if self.mode == "dense":
            return self.data
        elif self.mode == "count":
            # interpret counts as repeated spikes
            return torch.clamp(self.data, min=0).float()
        elif self.mode == "event":
            if shape is None:
                raise ValueError("Need shape for event->dense")
            dense = torch.zeros(shape, dtype=torch.float32)
            for (b,t,d) in self.data:
                dense[b,t,d] = 1.0
            return dense
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def to_count(self):
        if self.mode == "count":
            return self.data
        elif self.mode == "dense":
            return self.data.sum(dim=1, keepdim=True)  # sum over time
        elif self.mode == "event":
            raise NotImplementedError("event->count not implemented")
        else:
            raise ValueError

    def __repr__(self):
        return f"SpikeTensor(mode={self.mode}, shape={getattr(self.data,'shape',None)})"

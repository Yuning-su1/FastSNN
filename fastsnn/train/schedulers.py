import math

class WarmupCosine:
    """
    Linear warmup -> cosine decay.
    """
    def __init__(self, base_lr: float, warmup_steps: int, max_steps: int, min_lr: float = 0.0):
        self.base_lr = base_lr
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_steps = max_steps
        self.min_lr = min_lr

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        # cosine from warmup_steps -> max_steps
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self, optimizer, step: int):
        lr = self.lr_at(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

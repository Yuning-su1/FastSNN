
from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
from dataclasses import dataclass
from typing import Iterable, Optional

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 200
    log_every: int = 50

class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def fit(self, data_iter: Iterable[torch.Tensor], device: Optional[str] = None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        step = 0
        for batch in data_iter:
            x = batch.to(device)      # [B,T]
            y = x[:,1:].contiguous()
            inp = x[:,:-1].contiguous()
            logits = self.model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            step += 1
            if step % self.cfg.log_every == 0:
                print(f"step {step} loss {loss.item():.4f}")
            if step >= self.cfg.max_steps:
                break

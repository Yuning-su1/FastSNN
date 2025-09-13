from typing import Optional
import torch
import torch.nn as nn

from ..attention.linear import LinearAttention
from ..attention.sliding_window import SlidingWindowAttention
from ..ffn.pulse_ffn import PulseFFN
from .config import SNNConfig

class Block(nn.Module):
    def __init__(self, D: int, H: int, d_ff: int, cfg: SNNConfig, layer_id: int):
        super().__init__()
        # choose attention kind
        kind = cfg.attn_kind
        if kind == "hybrid_alt":
            kind = "linear" if (layer_id % 2 == 0) else "sliding"
        if kind == "linear":
            self.attn = LinearAttention(D, H, phi=cfg.attn_phi, dropout_p=cfg.dropout_p)
        elif kind == "sliding":
            self.attn = SlidingWindowAttention(D, H, window=cfg.sw_window, dropout_p=cfg.dropout_p)
        else:
            raise ValueError(f"Unknown attn_kind: {cfg.attn_kind}")

        self.norm1 = nn.LayerNorm(D)
        self.ffn = PulseFFN(D, d_ff,
                            neuron_type=cfg.neuron_type,
                            tau=cfg.neuron_tau, theta=cfg.neuron_theta,
                            theta_learnable=cfg.neuron_theta_learnable,
                            ste_tau=cfg.neuron_ste_tau,
                            dropout_p=cfg.dropout_p)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x

class SNNLanguageModel(nn.Module):
    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg
        D, H, L = cfg.d_model, cfg.n_heads, cfg.n_layers
        self.tok = nn.Embedding(cfg.vocab_size, D)
        self.blocks = nn.ModuleList([Block(D, H, cfg.d_ff, cfg, i) for i in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)
        if cfg.tie_lm_head:
            self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

def build_model_from_config(cfg: SNNConfig) -> nn.Module:
    return SNNLanguageModel(cfg)


from __future__ import annotations
import torch, torch.nn as nn
from .config import SNNConfig
from fastsnn.attention.linear import LinearAttention
from fastsnn.attention.sliding_window import SlidingWindowAttention
from fastsnn.attention.hybrid import HybridMixAttention
from fastsnn.ffn.pulse_ffn import PulseFFN
from fastsnn.core.spike_tensor import to_count_if_spike, wrap_like_input
class Block(nn.Module):
    def __init__(self, cfg: SNNConfig, layer_id: int):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        if cfg.attn_kind == 'linear':
            self.attn = LinearAttention(D, H, dropout_p=cfg.dropout)
        elif cfg.attn_kind == 'sliding':
            self.attn = SlidingWindowAttention(D, H, window=cfg.window, dropout_p=cfg.dropout)
        elif cfg.attn_kind == 'hybrid_alt':
            include_softmax = (layer_id % 6 == 0)
            self.attn = HybridMixAttention(D, H, window=cfg.window, include_softmax=include_softmax, dropout_p=cfg.dropout)
        else:
            raise ValueError(cfg.attn_kind)
        self.ffn  = PulseFFN(D, cfg.d_ff, dropout=cfg.dropout)

    def forward(self, x, kv_state=None, incremental=False):
        x_in = x
        y, kv_state = self.attn(self.norm1(x), kv_state=kv_state, incremental=incremental)
        x = to_count_if_spike(x) + to_count_if_spike(y)  
        y = self.ffn(self.norm2(x))
        x = x + to_count_if_spike(y)
        x = wrap_like_input(x, x_in, kind="count")
        return x, kv_state

class SNNLanguageModel(nn.Module):
    def __init__(self, cfg: SNNConfig):
        super().__init__()
        D = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, D)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)
        if cfg.tie_lm_head:
            self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor):
        kv = None
        x = self.tok(idx)
        for i,blk in enumerate(self.blocks):
            x, kv = blk(x, kv_state=kv, incremental=False)
        x = self.ln_f(x)
        return self.head(x)

def build_model_from_config(cfg: SNNConfig) -> nn.Module:
    return SNNLanguageModel(cfg)

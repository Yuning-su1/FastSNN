import torch, torch.nn as nn
from fastsnn.attention.linear import LinearAttention
from fastsnn.ffn.pulse_ffn import PulseFFN

class Block(nn.Module):
    def __init__(self, D, H, d_ff, attn_cfg, neuron_cfg, layer_id=0):
        super().__init__()
        kind = attn_cfg["kind"]
        if kind == "hybrid_alt":
            # 交替：奇数层 linear，偶数层 swa
            kind = "linear" if (layer_id % 2 == 0) else "sliding"

        if kind == "linear":
            self.attn = LinearAttention(D, H, phi_kind=attn_cfg["phi"])
        elif kind == "sliding":
            self.attn = SlidingWindowAttention(D, H, window=attn_cfg["sw_window"])
        else:
            raise ValueError(f"Unknown attention kind: {kind}")

        self.ln1  = nn.LayerNorm(D)
        self.ffn  = PulseFFN(D, d_ff, neuron_cfg)
        self.ln2  = nn.LayerNorm(D)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TinySNNLM(nn.Module):
    def __init__(self, cfg, vocab_size=50257):
        super().__init__()
        D, H, L, dff = cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["d_ff"]
        self.tok = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList([Block(D,H,dff,cfg["attn"],cfg["neuron"]) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)
    def forward(self, idx):
        # idx: [B,T]
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

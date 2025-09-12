import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window=128):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.window = window

    def forward(self, x):
        # x: [B,T,D]
        B,T,D = x.shape
        q = rearrange(self.Wq(x), "b t (h d) -> b h t d", h=self.h)
        k = rearrange(self.Wk(x), "b t (h d) -> b h t d", h=self.h)
        v = rearrange(self.Wv(x), "b t (h d) -> b h t d", h=self.h)

        # 局部窗口注意力：对每个 t，只看 [t-w, t]
        w = self.window
        attn_out = torch.empty_like(q)
        for t in range(T):
            s = max(0, t - w)
            q_t = q[:, :, t:t+1, :]               # [B,H,1,d]
            k_sw = k[:, :, s:t+1, :]              # [B,H,L,d]
            v_sw = v[:, :, s:t+1, :]
            scores = torch.einsum("b h 1 d, b h L d -> b h 1 L", q_t, k_sw) / (self.dk ** 0.5)
            probs = scores.softmax(dim=-1)
            out = torch.einsum("b h 1 L, b h L d -> b h 1 d", probs, v_sw)
            attn_out[:, :, t:t+1, :] = out

        out = rearrange(attn_out, "b h t d -> b t (h d)")
        return self.Wo(out)

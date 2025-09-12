import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

def phi(x, kind="relu"):
    if kind == "relu":
        return F.relu(x) + 1e-6
    elif kind == "softplus":
        return F.softplus(x) + 1e-6
    else:
        raise ValueError

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, phi_kind="relu"):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.phi_kind = phi_kind

    def forward(self, x):
        # x: [B,T,D]
        B,T,D = x.shape
        q = rearrange(self.Wq(x), "b t (h d) -> b h t d", h=self.h)
        k = rearrange(self.Wk(x), "b t (h d) -> b h t d", h=self.h)
        v = rearrange(self.Wv(x), "b t (h d) -> b h t d", h=self.h)

        qf, kf = phi(q, self.phi_kind), phi(k, self.phi_kind)

        # 递推累积（测试版：用前缀和近似代替 online scan）
        kv = torch.einsum("b h t d, b h t e -> b h d e", kf, v)     # Σ_t k_t v_t
        z  = torch.einsum("b h t d, b h t e -> b h d e", kf, torch.ones_like(v[...,:1]))  # Σ_t k_t * 1
        out = torch.einsum("b h t d, b h d e -> b h t e", qf, kv) / \
              (torch.einsum("b h t d, b h d e -> b h t e", qf, z) + 1e-6)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.Wo(out)
from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn

from .config import SNNConfig
from .model_from_config import SNNLanguageModel
from ..attention.linear import LinearAttention
from ..attention.sliding_window import SlidingWindowAttention

# hybrid attention is optional; if not present, we'll gracefully fallback.
try:
    from ..attention.hybrid import HybridMixAttention   # (linear + window [+ softmax])
    _HYBRID_AVAILABLE = True
except Exception:
    _HYBRID_AVAILABLE = False

from ..ffn.pulse_ffn import PulseFFN

# -----------------------------
# helpers: soft introspection
# -----------------------------

def _find_module_by_names(model: nn.Module, names: List[str]) -> Optional[nn.Module]:
    for n, m in model.named_modules():
        base = n.split(".")[-1].lower()
        if base in names:
            return m
    return None

def _maybe_linear_weight(m: nn.Module) -> Optional[torch.Tensor]:
    return getattr(m, "weight", None)

def _get_hf_attn_qkv(ann_block: nn.Module) -> Tuple[Optional[nn.Linear], Optional[nn.Linear], Optional[nn.Linear], Optional[nn.Linear]]:
    """
    Try to resolve Q/K/V/WO in common HF architectures:
      - LLaMA/Qwen-like: attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
      - GPT-NeoX-like:   attention.query_key_value (packed), attention.dense
      - GPT2-like:       c_attn (packed), c_proj
    Returns (Wq, Wk, Wv, Wo) as Linear layers (or None if not resolvable).
    """
    # style A: q_proj/k_proj/v_proj/o_proj
    q = getattr(ann_block, "q_proj", None) or _find_module_by_names(ann_block, ["q_proj"])
    k = getattr(ann_block, "k_proj", None) or _find_module_by_names(ann_block, ["k_proj"])
    v = getattr(ann_block, "v_proj", None) or _find_module_by_names(ann_block, ["v_proj"])
    o = getattr(ann_block, "o_proj", None) or _find_module_by_names(ann_block, ["o_proj"])
    if all([q, k, v, o]):
        return q, k, v, o

    # style B: packed QKV
    qkv = _find_module_by_names(ann_block, ["c_attn", "query_key_value"])
    o   = _find_module_by_names(ann_block, ["c_proj", "dense"])
    if qkv is not None and hasattr(qkv, "weight") and o is not None:
        # We will slice later; return qkv in q-slot, mark k,v as None to indicate packed
        return qkv, None, None, o

    return None, None, None, None

def _slice_qkv_packed(weight: torch.Tensor, bias: Optional[torch.Tensor], d_model: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Split a packed [out: 3*d_model, in: d_model] into (Wq, Wk, Wv[, bq, bk, bv]).
    """
    out, in_f = weight.shape
    assert out % 3 == 0, "packed QKV must have 3*d_model rows"
    d = out // 3
    Wq, Wk, Wv = weight[:d, :].clone(), weight[d:2*d, :].clone(), weight[2*d:, :].clone()
    bq = bk = bv = None
    if bias is not None:
        bq, bk, bv = bias[:d].clone(), bias[d:2*d].clone(), bias[2*d:].clone()
    return Wq, Wk, Wv, bq, bk, bv

def _get_hf_ffn(ann_block: nn.Module) -> Tuple[Optional[nn.Linear], Optional[nn.Linear]]:
    """
    Return (W1, W2) Linear layers of MLP/FFN if possible.
    Common patterns:
      - up_proj/down_proj
      - fc_in/fc_out
      - c_fc/c_proj
    """
    W1 = _find_module_by_names(ann_block, ["up_proj", "c_fc", "fc_in", "gate_proj"])
    W2 = _find_module_by_names(ann_block, ["down_proj", "c_proj", "fc_out"])
    return W1, W2

# -----------------------------
# MoE upcycling (optional)
# -----------------------------

class _TopKRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, k: int):
        super().__init__()
        self.w = nn.Linear(d_model, n_experts, bias=False)
        self.k = k

    def forward(self, x):  # x: [B,T,D]
        logits = self.w(x)                     # [B,T,E]
        probs  = torch.softmax(logits, dim=-1) # [B,T,E]
        topk = torch.topk(probs, k=self.k, dim=-1)
        return probs, topk.indices, topk.values

class _PulseMoE(nn.Module):
    """
    Lightweight Pulse-FFN MoE wrapper used only for upcycling at conversion time.
    - experts: N copies of PulseFFN (cloned from dense FFN)
    - shared experts: S copies always-on (S=1 typical)
    - router: Top-k gating
    - scaling: keep post-upcycling activation scale stable
    """
    def __init__(self, base_ffn: PulseFFN, n_experts: int, top_k: int, n_shared: int = 1,
                 scaling_factor: Optional[float] = None):
        super().__init__()
        assert top_k <= n_experts, "top_k must be <= n_experts"
        self.d_model = base_ffn.fc2.out_features
        self.experts = nn.ModuleList([self._clone_ffn(base_ffn) for _ in range(n_experts)])
        self.shared  = nn.ModuleList([self._clone_ffn(base_ffn) for _ in range(n_shared)]) if n_shared > 0 else None
        self.router  = _TopKRouter(self.d_model, n_experts, top_k)
        # default scaling from paper intuition: keep E[activation] similar
        if scaling_factor is None:
            # heuristic: 1 / sqrt(S + k/N)
            scaling_factor = 1.0 / math.sqrt((n_shared or 0) + top_k / max(1, n_experts))
        self.register_buffer("scale", torch.tensor(float(scaling_factor)))

    @staticmethod
    def _clone_ffn(ffn: PulseFFN) -> PulseFFN:
        import copy
        clone = copy.deepcopy(ffn)
        return clone

    def forward(self, x):  # [B,T,D]
        probs, idx, vals = self.router(x)  # probs:[B,T,E], idx/vals:[B,T,k]
        B,T,D = x.shape
        acc = torch.zeros_like(x)

        # shared experts (always on)
        if self.shared is not None:
            for s in self.shared:
                acc = acc + s(x)

        # routed experts
        # naive loop for clarity (optimize later)
        for b in range(B):
            for t in range(T):
                xt = x[b:b+1, t:t+1, :]        # [1,1,D]
                for j in range(idx.size(-1)):
                    e_idx = int(idx[b, t, j])
                    w = vals[b, t, j]          # scalar gate
                    acc[b:b+1, t:t+1, :] += w * self.experts[e_idx](xt)

        return self.scale * acc

# -----------------------------
# main conversion
# -----------------------------

def _build_snn_from_ann_config(ann_model: nn.Module, cfg: Optional[SNNConfig]) -> SNNLanguageModel:
    if cfg is not None:
        return SNNLanguageModel(cfg)

    # Infer minimal config from HF model if available
    vocab = 50257
    hidden = 256
    heads  = 4
    layers = 4
    d_ff   = 1024
    if hasattr(ann_model, "config"):
        hfc = ann_model.config
        vocab = getattr(hfc, "vocab_size", vocab)
        hidden = getattr(hfc, "hidden_size", hidden)
        heads  = getattr(hfc, "num_attention_heads", heads)
        layers = getattr(hfc, "num_hidden_layers", layers)
        d_ff   = getattr(hfc, "intermediate_size", d_ff)
    guess = SNNConfig(vocab_size=vocab, d_model=hidden, n_heads=heads, n_layers=layers, d_ff=d_ff)
    return SNNLanguageModel(guess)

def _transplant_embed_and_head(ann: nn.Module, snn: SNNLanguageModel):
    # embeddings
    try:
        inp = ann.get_input_embeddings().weight.data
        snn.tok.weight.data[:inp.shape[0], :inp.shape[1]].copy_(inp)
    except Exception:
        pass
    # LM head
    try:
        out = ann.get_output_embeddings().weight.data
        snn.head.weight.data[:out.shape[0], :out.shape[1]].copy_(out)
    except Exception:
        pass

def _init_linear_like(dst: nn.Linear, w: torch.Tensor, b: Optional[torch.Tensor] = None):
    with torch.no_grad():
        if dst.weight.shape == w.shape:
            dst.weight.copy_(w)
        else:
            # fallback: fan-in compatible copy
            rows = min(dst.weight.shape[0], w.shape[0])
            cols = min(dst.weight.shape[1], w.shape[1])
            dst.weight[:rows, :cols].copy_(w[:rows, :cols])
        if b is not None and getattr(dst, "bias", None) is not None:
            if dst.bias.shape == b.shape:
                dst.bias.copy_(b)
            else:
                dst.bias[:min(dst.bias.shape[0], b.shape[0])].copy_(b[:min(dst.bias.shape[0], b.shape[0])])

def _convert_block(ann_block: nn.Module, snn_block: nn.Module, cfg: SNNConfig, layer_id: int):
    """
    Map attention & FFN weights:
    - Attention: copy Q/K/V/WO; for packed QKV split before copy.
    - FFN: copy W1/W2 into PulseFFN.fc1/fc2.
    - Keep LayerNorm small/new params intact (low-rank/small, no heavy reinit).
    """
    # ---- Attention ----
    Wq, Wk, Wv, Wo = _get_hf_attn_qkv(ann_block)
    # resolve SNN attention impl
    attn = snn_block.attn

    if Wq is not None and Wk is None and Wv is None:
        # packed QKV
        W = Wq.weight
        b = getattr(Wq, "bias", None)
        d_model = W.shape[1]
        Wq_t, Wk_t, Wv_t, bq, bk, bv = _slice_qkv_packed(W, b, d_model=d_model)
        if hasattr(attn, "Wq") and hasattr(attn, "Wk") and hasattr(attn, "Wv"):
            _init_linear_like(attn.Wq, Wq_t, bq)
            _init_linear_like(attn.Wk, Wk_t, bk)
            _init_linear_like(attn.Wv, Wv_t, bv)
    else:
        if Wq is not None and hasattr(Wq, "weight") and hasattr(attn, "Wq"):
            _init_linear_like(attn.Wq, Wq.weight, getattr(Wq, "bias", None))
        if Wk is not None and hasattr(Wk, "weight") and hasattr(attn, "Wk"):
            _init_linear_like(attn.Wk, Wk.weight, getattr(Wk, "bias", None))
        if Wv is not None and hasattr(Wv, "weight") and hasattr(attn, "Wv"):
            _init_linear_like(attn.Wv, Wv.weight, getattr(Wv, "bias", None))

    if Wo is not None and hasattr(Wo, "weight") and hasattr(attn, "Wo"):
        _init_linear_like(attn.Wo, Wo.weight, getattr(Wo, "bias", None))

    # Note: LinearAttention requires non-negative kernel φ; this is in impl, not in weights.

    # ---- FFN ----
    W1, W2 = _get_hf_ffn(ann_block)
    if isinstance(snn_block.ffn, PulseFFN):
        if W1 is not None and hasattr(W1, "weight"):
            _init_linear_like(snn_block.ffn.fc1, W1.weight, getattr(W1, "bias", None))
        if W2 is not None and hasattr(W2, "weight"):
            _init_linear_like(snn_block.ffn.fc2, W2.weight, getattr(W2, "bias", None))

def _maybe_upcycle_moe(snn_block: nn.Module, moe_cfg: Optional[Dict[str, Any]]):
    """
    Replace block.ffn with MoE upcycled wrapper if configured.
    moe_cfg: {"enabled": True, "n_experts": N, "top_k": k, "n_shared": S, "scaling": float|None}
    """
    if not moe_cfg or not moe_cfg.get("enabled", False):
        return
    assert isinstance(snn_block.ffn, PulseFFN), "MoE upcycling expects PulseFFN base"
    N = int(moe_cfg.get("n_experts", 8))
    k = int(moe_cfg.get("top_k", 1))
    S = int(moe_cfg.get("n_shared", 1))
    scaling = moe_cfg.get("scaling", None)
    snn_block.ffn = _PulseMoE(snn_block.ffn, n_experts=N, top_k=k, n_shared=S, scaling_factor=scaling)

def convert_ann_to_snn(
    ann_model: nn.Module,
    cfg: Optional[SNNConfig] = None,
    *,
    attention_target: str = "hybrid_alt",   # "linear" | "sliding" | "hybrid_alt" | "hybrid_mix"
    phi: str = "relu",                      # non-neg kernel for linear
    sw_window: int = 128,
    long_context_target: Optional[int] = None,  # e.g., 8192/32768 during conversion
    moe_upcycling: Optional[Dict[str, Any]] = None,  # see _maybe_upcycle_moe
    adaptive_threshold_k: Optional[float] = None,    # sINT neuron hyperparam hook
) -> Tuple[SNNLanguageModel, Dict[str, Any]]:
    """
    Convert pretrained ANN (softmax attention) to an SNN skeleton with reused projections.

    Returns:
      snn_model, meta  (meta contains conversion hints: non-neg kernel, long-context plan, etc.)
    """
    snn = _build_snn_from_ann_config(ann_model, cfg)

    # Attach global choices to config-like fields so downstream stays aware:
    # - attention kind and φ
    for i, blk in enumerate(snn.blocks):
        # if hybrid not available, fallback to alternating linear/sliding
        if attention_target == "hybrid_mix" and not _HYBRID_AVAILABLE:
            kind = "hybrid_alt"
        else:
            kind = attention_target

        # mutate attn module if needed
        D = blk.ln1.normalized_shape[0]
        H = blk.attn.h if hasattr(blk.attn, "h") else getattr(snn, "n_heads", 4)

        if kind == "linear":
            blk.attn = LinearAttention(D, H, phi_kind=phi)
        elif kind == "sliding":
            blk.attn = SlidingWindowAttention(D, H, window=sw_window)
        elif kind == "hybrid_alt":
            # odd/even alternating: linear/sliding
            blk.attn = LinearAttention(D, H, phi_kind=phi) if (i % 2 == 0) \
                       else SlidingWindowAttention(D, H, window=sw_window)
        elif kind == "hybrid_mix" and _HYBRID_AVAILABLE:
            blk.attn = HybridMixAttention(D, H, phi_kind=phi, window=sw_window)
        else:
            raise ValueError(f"Unknown attention_target: {attention_target}")

        # set neuron hyperparam (adaptive threshold k) if provided
        if adaptive_threshold_k is not None and hasattr(blk.ffn, "neuron"):
            # You can store k into neuron or meta; here we attach as attribute for training to read.
            setattr(blk.ffn.neuron, "k_adaptive", float(adaptive_threshold_k))

    # transplant embeddings / head
    _transplant_embed_and_head(ann_model, snn)

    # per-block transplant + optional MoE
    ann_blocks = [m for m in ann_model.modules() if isinstance(m, nn.Module) and len(list(m.children())) > 0]
    # heuristic: zip shortest
    for i, snn_blk in enumerate(snn.blocks):
        ann_blk = ann_blocks[i] if i < len(ann_blocks) else ann_model
        _convert_block(ann_blk, snn_blk, snn.cfg if hasattr(snn, "cfg") else SNNConfig(), i)
        _maybe_upcycle_moe(snn_blk, moe_upcycling)

    # meta info for trainer
    meta = {
        "attention_target": attention_target,
        "phi": phi,
        "sw_window": sw_window,
        "non_negative_kernel": True,   # critical for linear attention stability
        "reuse_qkvwo": True,
        "reuse_ffn": True,
        "minimal_new_params": True,
        "conversion_learnings": [
            "Keep new params small/low-rank (norm/gates).",
            "Warmup LR; consider full-parameter training if feasible.",
        ],
    }

    # conversion-phase long-context extension (trainer should apply LR warmup)
    if long_context_target is not None:
        meta["long_context_plan"] = {
            "target_seq_len": int(long_context_target),
            "do_in_conversion": True,
            "lr_warmup_ratio": 0.2
        }

    # store adaptive-threshold hint (k)
    if adaptive_threshold_k is not None:
        meta["adaptive_threshold_k"] = float(adaptive_threshold_k)

    # MoE upcycling plan
    if moe_upcycling and moe_upcycling.get("enabled", False):
        meta["moe_upcycling"] = {
            "n_experts": int(moe_upcycling.get("n_experts", 8)),
            "top_k": int(moe_upcycling.get("top_k", 1)),
            "n_shared": int(moe_upcycling.get("n_shared", 1)),
            "scaling": moe_upcycling.get("scaling", None),
            "note": "Experts cloned from dense FFN; router randomized; shared expert always on."
        }

    return snn, meta

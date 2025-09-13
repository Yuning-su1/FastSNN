# FastSNN — Build • Convert • Run (One-Click SNN SDK)

> **TL;DR**
>
> ```bash
> pip install -e .
> python oneclick.py
> ```
>
> Runs end-to-end: **Linear / Sliding / Hybrid attention SNN LM**, **ANN→SNN conversion demo**, and a **minimal training loop**.

---

## Why FastSNN?

* **One-click runnable**: `python oneclick.py` does forward smoke test → 10-step train → ANN→SNN demo.
* **ANN→SNN conversion (minimal, practical)**: swap `ReLU/GELU/SiLU` with **adaptive-threshold sINT neurons**; auto-wrap `Linear` to accept `[B,T,D]`.
* **Three attention flavors**:

  * **Linear** — O(T) state-space style with non-negative kernel & accumulators
  * **Sliding Window (SWA)** — local causal band mask
  * **Hybrid** — Linear + SWA (optionally Softmax) concatenated then projected
* **Unified SpikeTensor** abstraction: `dense / count / event` (training uses count proxy; inference can expand to event-driven).
* **Minimal trainer**: validates the end-to-end chain on CPU quickly.

---

## Design Philosophy (the “Framework”)

1. **Two-step decoupling**

   * **Training**: produce **spike counts (sINT)** in a **single step** via adaptive threshold + STE; optimize efficiently in dense mode.
   * **Inference**: optionally **expand counts to spike trains** when deploying to event-driven back-ends.

2. **Uniform contracts**

   * All sequence modules speak `[B, T, D]` and return the same shape (plus an optional `state` dict).
   * Attention modules share `forward(x, kv_state=None, incremental=False) → (y, new_state)`; FFN/neurons are pure `[B,T,D] → [B,T,D]`.

3. **Single source of truth: SpikeTensor**

   * Canonical representation for `dense / count / event` with clear, lossless conversions when possible.

4. **Progressive realism**

   * Start with a **minimal, robust** path (what you have now). Add biological realism / quantization / deployment hooks incrementally without breaking the core API.

5. **Composable registry**

   * A simple `Registry` lets you plug in neurons/blocks/heads uniformly; names become stable public API.

---

## Unified Architecture

```
                  ┌───────────────────────────────┐
   Build          │  SNNLanguageModel(cfg)        │
   (dense)  ───▶  │  Blocks: [Attention + FFN]×N  │  ───▶  Trainer / Inference
                  └───────────────────────────────┘
                            ▲           ▲
                            │           │
                     Neurons (sINT)   SpikeTensor

Attention options inside each block:
  • Linear (O(T) accumulators, non-negative φ)
  • Sliding Window (local causal band mask)
  • Hybrid (Linear + SWA [+ Softmax]) → concat → projection
```

**Module contracts**

```python
# Attention (shared)
y, state = attn(x, kv_state=None, incremental=False)  # x: [B,T,D], y: [B,T,D]

# FFN (pure)
y = ffn(x)  # [B,T,D] → [B,T,D]

# Neuron (training-time sINT)
y = AdaptiveThresholdNeuron(d_model)(x)  # [B,T,D] → [B,T,D] (spike-count proxy)
```

---

## Install

```bash
pip install -e .
```

> Requires: `torch`, `einops`, `pyyaml` (declared in `setup.py`).

---

## One-Click Run

```bash
python oneclick.py
```

It runs:

1. **(Optional) Install** editable package
2. **Forward smoke test**
3. **10-step minimal training**
4. **ANN→SNN conversion demo** (prints tensor shapes)

---

## ANN → SNN Conversion

**Minimal, safe mode**—convert standard MLP activations to sINT neurons while keeping Linear layers:

```python
from fastsnn.convert.ann2snn import convert_ann_to_snn, TinyANN

ann = TinyANN(in_dim=128, hidden=256, out_dim=10)     # reference ANN
snn = convert_ann_to_snn(ann, d_model_fallback=128)   # SNN-wrapped variant
```

* `ReLU/GELU/SiLU` → **AdaptiveThreshold** (training-time sINT via STE, time collapsed)
* `Linear` → wrapped to accept `[B,T,D]` (flatten T, run, restore)
* Unknown layers → left intact (conservative fallback)

> This is intentionally **minimal** to guarantee a **working path first**. You can later upgrade to faithful attention/normalization conversions, event-level simulators, and quantized deployment.

---

## Quick Model Build

```python
import torch
from fastsnn.builder.config import SNNConfig
from fastsnn.builder.model_from_config import build_model_from_config

cfg = SNNConfig(
    vocab_size=5000, d_model=128, n_heads=4, n_layers=2, d_ff=256,
    attn_kind='hybrid_alt', window=128, dropout=0.0
)
model = build_model_from_config(cfg)      # nn.Module
x = torch.randint(0, cfg.vocab_size, (2, 64))   # [B,T]
logits = model(x[:, :-1])                        # LM next-token logits
```

## Minimal Training (CPU-friendly)

```python
import torch
from fastsnn.builder.config import SNNConfig
from fastsnn.builder.model_from_config import build_model_from_config
from fastsnn.train.trainer import Trainer, TrainConfig

def synthetic_data(num_batches=20, B=2, T=64, V=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(num_batches):
        yield torch.randint(0, V, (B, T), generator=g)

cfg = SNNConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2, d_ff=256, attn_kind='hybrid_alt')
model = build_model_from_config(cfg)
trainer = Trainer(model, TrainConfig(max_steps=10, log_every=2, lr=1e-3))
trainer.fit(synthetic_data())
```

## What’s Inside

```
fastsnn/
  attention/  {linear.py, sliding_window.py, hybrid.py}
  builder/    {config.py, model_from_config.py}
  cli/        {main.py}
  convert/    {ann2snn.py}          # ANN→SNN (minimal, practical)
  core/       {registry.py, spike_tensor.py}
  ffn/        {pulse_ffn.py}
  neurons/    {adaptive_threshold.py, lif_sint.py, surrogate.py}
  train/      {trainer.py}
tests/
  test_train.py
oneclick.py
setup.py
```


## Implementation Notes

* **Adaptive-threshold sINT** (`neurons/adaptive_threshold.py`):
  computes a dynamic threshold (batch/feature statistics), uses **STE** to round to counts; returns `[B,T,D]` spike counts during training.

* **Linear Attention** (`attention/linear.py`):
  non-negative feature map `φ(·)`, recurrent accumulators `(kv_acc, z_acc)`, supports `incremental=True` decoding.
  *Dev tip:* if you ever use `einsum`, **never** write numeric dimensions (use letters, even when the size is 1).

* **Sliding Window Attention** (`attention/sliding_window.py`):
  causal band mask around the diagonal; window `w` trades compute for range.

* **Hybrid** (`attention/hybrid.py`):
  concatenate outputs of Linear + SWA (optionally Softmax), then project back to `d_model`.

* **SpikeTensor** (`core/spike_tensor.py`):
  fixed `to_dense()` in `dense` mode (no more `Ellipsis`), added robust `count→dense` expansion.



## Roadmap

* Faithful ANN→SNN mapping for Attention / Norm / Conv
* SpikeScope visualization (sparsity, energy proxy, spike stats)
* Deployment packaging: `.snn` format, W8A weight + low-bit KV, event simulators
* PyPI release and docs site


## FAQ

* **Does this run without GPU?**
  Yes. Everything here is CPU-friendly for the smoke test and short training.

* **Why is conversion minimal?**
  To guarantee a working end-to-end path. You can replace the neuron, gating, or quantization strategies later without breaking the public contract.

## License

Apache-2.0


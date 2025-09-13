# FastSNN — Build • Convert • Run

### An Open Toolchain for Practical Spiking Neural Networks

> **About**
> **Based on the algorithm proposed by the BIClab in the SpikingBrain project, we make an open chain of tools to enable users complete a SNN project conveniently.**
> FastSNN is a hands-on SDK that brings those ideas into an engineer-friendly pipeline you can run end-to-end: **build ANN → convert to SNN → train/evaluate → export/deploy**.

---

## 🚀 TL;DR (30 seconds)

```bash
pip install -e .
python oneclick.py
```

You’ll get:

* a forward pass of a **spiking language model** (Linear/SWA/Hybrid attention),
* a **10-step training smoke test** (CPU OK),
* and a **1-line ANN→SNN conversion demo** that actually runs.

---

## 🧠 Philosophical Framework

FastSNN follows three principles inspired by **SpikingBrain (BIClab)**:

1. **Two-step decoupling (training vs. inference)**

   * **Training** uses **instantaneous spike-count (sINT)** neurons with **adaptive thresholds**, collapsing the time dimension for stability and speed.
   * **Inference** can expand back to spike-time sequences without changing trained weights.

2. **Unified abstractions instead of monoliths**

   * A single **SpikeTensor** abstraction (`dense / count / event`)
   * A small, explicit **Registry** for pluggable neurons/attention/FFN
   * One **Builder/Config** path that constructs consistent models

3. **Hybrid compute: use what works**

   * Combine **Linear Attention** (state-space style, O(T)) with **Sliding-Window Attention (SWA)** for locality; optionally mix in Softmax on chosen layers.
   * Keep FFN simple but **spike-aware** (sINT injection) to validate end-to-end flow first.

---

## 🧩 Unified Architecture

```
                       ┌──────────────────────────────────────────────────────────┐
Data/Tokenization ───▶ │  ANN Backbones (MLP/Transformer blocks)                  │
                       └───────────────┬──────────────────────────────────────────┘
                                       │  (registry-driven components)
                                       ▼
                             ANN→SNN Conversion (training-time)
                        - Replace ReLU/GELU/SiLU → sINT w/ adaptive threshold
                        - Preserve Linear; wrap to [B,T,D] temporal shapes
                                       │
                                       ▼
                     SNN Blocks (Attention + FFN + Neurons + SpikeTensor)
                 - Linear Attention (state accumulators kv_acc / z_acc)
                 - Sliding-Window Attention (causal local band mask)
                 - Hybrid Mix (Linear + SWA [+ Softmax optional])
                 - PulseFFN (inject sINT into hidden activations)
                                       │
                                       ▼
                         Trainer / Evaluator  →  Export (.snn planned)
```

**Core abstractions**

* **SpikeTensor**: unifies training-time `count` and inference-time `event` spike views.
* **Registry**: clean plug-in points for neurons & attention; no hidden magic.
* **Builder/Config**: one place to define model size, heads, windows, etc.

---

## ✨ What’s Included (and runnable)

* **ANN→SNN Conversion (Minimal & Practical)**

  ```python
  from fastsnn.convert.ann2snn import convert_ann_to_snn, TinyANN
  ann = TinyANN(in_dim=128, hidden=256, out_dim=10)
  snn = convert_ann_to_snn(ann, d_model_fallback=128)   # one line
  ```

  Converts ReLU/GELU/SiLU → **Adaptive-Threshold sINT** (training-time), wraps Linear for `[B,T,D]`. Unknown modules are kept intact so you can iterate safely.

* **Attention Options**

  * **Linear**: non-negative kernel features + running state (`kv_acc`, `z_acc`), supports incremental decoding.
  * **SWA**: sliding, causal, memory-friendly locality via banded attention mask.
  * **Hybrid**: concatenates Linear + SWA (optionally Softmax on configured layers), then projects back.

* **Neuron family**

  * **AdaptiveThresholdNeuron** (sINT) and **LIF-sINT** with STE for differentiable count generation.

* **Trainer**

  * A tiny `Trainer.fit()` that proves backprop and loss improvement in minutes on CPU.

---

## 🔧 Quick Start

```bash
# one command to prove the whole pipeline
python oneclick.py
```

Under the hood this performs:

1. Editable install (optional).
2. Forward smoke test (default: Hybrid attention).
3. A 10-step mini-train.
4. ANN→SNN conversion demo (prints matching tensor shapes).

**Or use it programmatically:**

```python
import torch
from fastsnn.builder.config import SNNConfig
from fastsnn.builder.model_from_config import build_model_from_config

cfg = SNNConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2,
                d_ff=256, attn_kind='hybrid_alt')
model = build_model_from_config(cfg)

x = torch.randint(0, cfg.vocab_size, (2, 64))
logits = model(x[:, :-1])  # language modeling next-token logits
```

---

## 📚 Directory Layout

```
fastsnn/
  attention/  {linear.py, sliding_window.py, hybrid.py}
  builder/    {config.py, model_from_config.py}
  cli/        {main.py}
  convert/    {ann2snn.py}          # ANN→SNN conversion (minimal but real)
  core/       {registry.py, spike_tensor.py}
  ffn/        {pulse_ffn.py}
  neurons/    {adaptive_threshold.py, lif_sint.py, surrogate.py}
  train/      {trainer.py}
tests/
  test_train.py
oneclick.py
setup.py
```

---

## 🧪 Minimal Training (copy & run)

```python
import torch
from fastsnn.builder.config import SNNConfig
from fastsnn.builder.model_from_config import build_model_from_config
from fastsnn.train.trainer import Trainer, TrainConfig

def synthetic_data(num_batches=20, B=2, T=64, V=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(num_batches):
        yield torch.randint(0, V, (B, T), generator=g)

cfg = SNNConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                attn_kind='hybrid_alt')
model = build_model_from_config(cfg)
trainer = Trainer(model, TrainConfig(max_steps=10, log_every=2, lr=1e-3))
trainer.fit(synthetic_data())
```

---

## 🧪 ANN→SNN: What “Two-Step” Means Here

* **Training-time**: sINT neurons collapse the time dimension by counting spikes via **adaptive thresholds** (data-dependent), with **surrogate gradient (STE)** for backprop.
* **Inference-time**: you can expand to true **spike events** (planned utilities), preserving trained weights but switching the runtime to event-driven.

This mirrors the **instantaneous sINT** methodology from the SpikingBrain line of work and is the reason you can train fast while still gaining SNN deployment benefits later.

---

## 🧭 Troubleshooting (Common Pitfalls)

* **`einsum` subscripts error**: Make sure you **never** use digits in `einsum` subscripts. Even for a single-step time dimension, write `t` instead of `1` (already fixed in `attention/linear.py`).
* **All-zero or all-fire spikes**: tune the adaptive-threshold scaling `k` (in `AdaptiveThresholdNeuron`) or regularize activations to keep firing rates in a reasonable band.

---

## 🗺️ Roadmap

* Full ANN→SNN mapping (Attention/Norm/Conv temporalization)
* SpikeScope visualizations (firing rates, sparsity, energy proxies)
* Export format (`.snn`) and deployment recipes (W8A quant, low-bit KV)
* Docs site and examples gallery

---

## 🤝 Contributing

* Keep modules **small and composable** (respect the registry & SpikeTensor).
* Prefer **tests that prove an end-to-end path** (build→forward→train→convert).
* Submit PRs with a **short demo script** and expected outputs.

---

## 📜 License

Apache-2.0

---
# FastSNN

FastSNN is a developer-friendly SDK for building, training, and deploying Spiking Neural Networks (SNNs).  
Inspired by the algorithms introduced in BICLab’s *SpikingBrain*, FastSNN makes SNN research and applications accessible with concise APIs and standardized tooling.

## ✨ Key Features

- **Unified SNN File Format**  
  A `.snn` bundle (config + weights + meta + reports) for easy cross-platform deployment and model sharing.

- **Five Core APIs**  
  `build()`, `fit()`, `extend_context()`, `export()`, `serve()` cover the full development lifecycle.

- **SpikeTensor Abstraction**  
  A unified representation of spikes (dense, sparse, or integer-count), ensuring ANN-style interfaces while preserving SNN semantics.

- **Coder → NeuronCell Chain**  
  Input encoding, neuron dynamics, and layer composition follow a clean bottom-up hierarchy.

- **SpikeScope Visualization**  
  Built-in tools to inspect raster plots, firing rates, sparsity heatmaps, and memory/throughput scaling, making SNN behavior interpretable.

## 🚀 Quickstart

```python
from fastsnn import FastSNN

# 1. Build a model
model = FastSNN.build(arch="snn-tiny", attn="hybrid_alt", neuron="lif_sint")

# 2. Train with HuggingFace datasets
trainer = FastSNN.fit(model, dataset="wikitext/wikitext-2-raw-v1", seq_len=2048)

# 3. Extend context length (2k → 8k tokens)
FastSNN.extend_context(model, trainer, new_seq_len=8192)

# 4. Visualize spikes and sparsity
FastSNN.scope(model).report("runs/exp_demo")

# 5. Export and serve
FastSNN.export(model, "artifacts/my_model.snn")
FastSNN.serve("artifacts/my_model.snn", port=8000)
````

## 🧩 Philosophy

FastSNN is built on three unified abstractions:

1. **SpikeTensor** – the universal spike container
2. **Coder** – input/output transformation into spikes
3. **NeuronCell** – minimal computational unit for spike dynamics

This ensures developers can use SNNs with the same ease as ANN frameworks, while still benefiting from event-driven sparsity and long-sequence efficiency.

## 📊 SpikeScope

Visual diagnostics are first-class citizens in FastSNN:

* Spike raster plots
* Membrane potential traces
* Sparsity and firing-rate curves
* Memory/throughput scaling vs. sequence length

## 📦 Standardized Bundles

A FastSNN `.snn` model bundle contains:

```
model.snn/
  config.json      # model & training config
  weights.safetensors
  meta.json        # tokenizer, training milestones
  spike.report/    # visual & numerical diagnostics
```

## 📜 License

Apache 2.0
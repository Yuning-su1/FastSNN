# FastSNN

FastSNN is a developer-friendly SDK for building, training, and deploying **Spiking Neural Network (SNN) models**.  
Inspired by the algorithms introduced by BICLab’s *SpikingBrain* project, FastSNN provides a simple, consistent interface—bringing SNN development closer to the usability of mainstream deep learning frameworks.

## ✨ Key Features
- **One-line SNN building and training**  
  Create, train, or convert from ANN to SNN with concise APIs or CLI commands.

- **Unified SNN file format (.snn)**  
  Standardized model packaging for cross-platform deployment and long-term storage.

- **End-to-end workflow coverage**  
  From model definition, training, context extension, evaluation, to deployment—all in one toolkit.

- **SpikeTensor abstraction**  
  A core data structure unifying dense activations, spike counts (sINT), and event-based streams.  
  Built on the `Coder → NeuronCell` pipeline, it ensures ANN-like interfaces while retaining SNN semantics.

- **SpikeScope visualization**  
  A built-in tool for raster plots, sparsity curves, membrane potential traces, and throughput profiling.  
  Designed to make SNN evaluation interpretable and reproducible.

## 🚀 Quick Example
```python
from fastsnn import FastSNN

# Build a tiny SNN model
model = FastSNN.build(arch="snn-tiny", d_model=256, n_layers=4, n_heads=4)

# Train on WikiText
trainer = FastSNN.fit(model, dataset="wikitext/wikitext-2-raw-v1", seq_len=2048, max_steps=1000)

# Extend context length (2k → 8k)
FastSNN.extend_context(model, trainer, new_seq_len=8192)

# Generate visualization report
FastSNN.scope(model).report("runs/exp1")

# Export and serve
FastSNN.export(model, "artifacts/my_model.snn")
FastSNN.serve("artifacts/my_model.snn", port=8000)
````

## 📂 File Structure

```
fastsnn/
  core/         # Shared abstractions (SpikeTensor, Config, Registry)
  neurons/      # LIF-sINT, AdEx-sINT, surrogate gradients
  attention/    # Linear, Sliding Window, Hybrid
  ffn/          # Pulse FFN, MoE FFN
  builder/      # Model constructors, ANN→SNN converters
  trainer/      # Training loop, context extension, schedulers
  data/         # Dataset adapters, coders, streamers
  scope/        # SpikeScope visualization utilities
  deploy/       # Export, .snn bundling, REST serving
  cli/          # FastSNN CLI entrypoints
```

## 📜 License

Apache 2.0 – open for both research and commercial use.

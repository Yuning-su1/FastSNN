import os, time, json
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
from .hooks import TensorTap

class ScopeSession:
    """Holds raw traces captured during a SpikeScope run."""
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.meta: Dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": 0
        }
        self.traces: Dict[str, List] = {}
        os.makedirs(run_dir, exist_ok=True)

    def log_trace(self, name: str, value):
        self.traces.setdefault(name, []).append(value)

    def save_json(self, name: str, obj):
        with open(os.path.join(self.run_dir, name), "w") as f:
            json.dump(obj, f, indent=2)

    def save(self):
        self.save_json("meta.json", self.meta)
        self.save_json("traces.json", self.traces)

class SpikeScope:
    """
    Usage:
      with SpikeScope(model, run_dir="runs/exp1") as scope:
          logits = model(x)
      scope.report()  # generate default plots
    """
    def __init__(self, model: torch.nn.Module, run_dir: str = "runs/exp"):
        self.model = model
        self.run_dir = run_dir
        self.session = ScopeSession(run_dir)
        self.tap = TensorTap()
        self._attached = False

    def attach_default(self):
        """Attach default hooks: capture sINT activation tensors and per-layer sparsity."""
        if self._attached: return
        # Heuristics: tap any submodule with attribute name containing 'neuron' or 'ffn'
        for name, mod in self.model.named_modules():
            lname = name.lower()
            if ("neuron" in lname) or ("sint" in lname):
                self.tap.tap(mod, f"{name}:sint", reduce="none")
                self.tap.tap(mod, f"{name}:sparsity", reduce="sparsity")
        self._attached = True

    def __enter__(self):
        self.attach_default()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()
        self.flush()

    def detach(self):
        self.tap.close()

    def flush(self):
        # move tap.cache → session.traces
        for k, v in self.tap.cache.items():
            for item in v:
                self.session.log_trace(k, item)
        self.tap.clear()

    # ----- plotting -----

    def _plot_sparsity_curve(self):
        # Aggregate per-layer sparsity to a single curve (mean over hooks each step)
        # traces entries with ":sparsity"
        steps = 0
        layer_means = {}
        for k, arr in self.session.traces.items():
            if k.endswith(":sparsity"):
                steps = max(steps, len(arr))
                layer_means[k] = arr
        if steps == 0: return
        import numpy as np
        xs = np.arange(steps)
        ys = []
        for t in range(steps):
            vals = []
            for k, arr in layer_means.items():
                if t < len(arr): vals.append(arr[t])
            if vals:
                ys.append(float(np.mean(vals)))
            else:
                ys.append(float("nan"))
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("step"); plt.ylabel("sparsity (ratio of zeros)")
        plt.title("Mean Layer Sparsity over Steps")
        out = os.path.join(self.run_dir, "sparsity.png")
        plt.savefig(out); plt.close()
        return out

    def _plot_raster_sample(self, max_steps=4, max_neurons=256):
        # Pick first captured sINT tensor and draw a tiny raster: time on x, neuron on y
        import numpy as np
        tens = None
        name = None
        for k, arr in self.session.traces.items():
            if k.endswith(":sint") and len(arr)>0 and isinstance(arr[0], torch.Tensor):
                name = k; tens = arr[0]; break
        if tens is None: return None
        # tens: [B,T,D] (from our sINT neuron)
        t = tens
        if t.ndim==3:
            b,tlen,d = t.shape
            mat = (t[0].ne(0)).float().T  # [D,T]; 1 if active
            mat = mat[:max_neurons, :min(tlen, 512)]
            plt.figure(figsize=(8,4))
            plt.imshow(mat, aspect="auto", interpolation="nearest")
            plt.xlabel("time"); plt.ylabel("neurons")
            plt.title(f"Raster (active=1) from {name}")
            out = os.path.join(self.run_dir, "raster.png")
            plt.savefig(out); plt.close()
            return out
        return None

    def _plot_throughput_dummy(self):
        # simple placeholder: steps/sec from meta
        if self.session.meta.get("steps", 0) <= 0: return None
        plt.figure()
        plt.bar([0], [self.session.meta["steps"]])
        plt.xticks([0], ["steps"])
        plt.ylabel("count")
        plt.title("Steps (placeholder for throughput)")
        out = os.path.join(self.run_dir, "throughput.png")
        plt.savefig(out); plt.close()
        return out

    def report(self):
        self.session.save()
        a = self._plot_sparsity_curve()
        b = self._plot_raster_sample()
        c = self._plot_throughput_dummy()
        # write a small HTML
        html = ["<h1>SpikeScope Report</h1>"]
        for title, path in [("Sparsity", a), ("Raster", b), ("Throughput", c)]:
            if path:
                html += [f"<h2>{title}</h2>", f'<img src="{os.path.basename(path)}" width="720">']
        with open(os.path.join(self.run_dir, "report.html"), "w") as f:
            f.write("\n".join(html))
        return os.path.join(self.run_dir, "report.html")

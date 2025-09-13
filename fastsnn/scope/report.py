import os, json

class ScopeReport:
    """Tiny helper to read saved scope session and return paths for UI or CI artifacts."""
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.meta = self._read("meta.json")
        self.traces = self._read("traces.json")

    def _read(self, name):
        p = os.path.join(self.run_dir, name)
        return json.load(open(p)) if os.path.exists(p) else {}

    def list_artifacts(self):
        files = []
        for f in ["report.html", "sparsity.png", "raster.png", "throughput.png"]:
            p = os.path.join(self.run_dir, f)
            if os.path.exists(p): files.append(p)
        return files

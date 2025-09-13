import os, json, shutil, torch
from typing import Dict, Any, Optional

def export_snn(model, cfg: Dict[str, Any], out_dir: str = "artifacts/model.snn",
               meta: Optional[Dict[str, Any]] = None,
               weights_name: str = "weights.safetensors",
               scope_dir: Optional[str] = None):
    """
    Export a .snn bundle:
      out_dir/
        config.json
        weights.safetensors
        meta.json
        spike.report/   (optional, copy from scope_dir)
    """
    os.makedirs(out_dir, exist_ok=True)
    # weights
    weights_path = os.path.join(out_dir, weights_name)
    torch.save(model.state_dict(), weights_path)
    # config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # meta
    meta = meta or {}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # attach SpikeScope report
    if scope_dir and os.path.exists(scope_dir):
        dst = os.path.join(out_dir, "spike.report")
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(scope_dir, dst)
    return out_dir

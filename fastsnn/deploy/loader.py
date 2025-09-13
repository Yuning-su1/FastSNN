import os, json, torch
from typing import Any, Dict

def load_snn(bundle_dir: str):
    """
    Return (config:dict, state_dict:OrderedDict, meta:dict)
    """
    cfg = json.load(open(os.path.join(bundle_dir, "config.json")))
    meta = json.load(open(os.path.join(bundle_dir, "meta.json")))
    # try two names
    wpath = None
    for name in ["weights.safetensors", "weights.pt"]:
        p = os.path.join(bundle_dir, name)
        if os.path.exists(p): wpath = p; break
    if wpath is None: raise FileNotFoundError("weights not found in bundle")
    state = torch.load(wpath, map_location="cpu")
    return cfg, state, meta

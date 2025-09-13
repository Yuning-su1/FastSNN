import os, json
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Lazy imports for torch/transformers to avoid mandatory dependency when not serving
def _lazy_import():
    import torch
    from transformers import AutoTokenizer
    return torch, AutoTokenizer

class GenPayload(BaseModel):
    text: str
    top_k: int = 5

def serve_snn(bundle_dir: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Start a simple REST service:
      POST /generate {"text": "...", "top_k": 5}
    """
    from .loader import load_snn
    torch, AutoTokenizer = _lazy_import()

    cfg, state, meta = load_snn(bundle_dir)
    vocab = meta.get("tokenizer", "gpt2")
    tok = AutoTokenizer.from_pretrained(vocab)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # user must provide TinySNNLM or FastSNN.build
    try:
        from fastsnn import FastSNN  # expect your SDK exposes this
        model = FastSNN.build(**cfg)
    except Exception:
        # fallback: try a known tiny model class
        from fastsnn.builder.model_from_config import TinySNNLM
        model = TinySNNLM(cfg, vocab_size=tok.vocab_size)

    model.load_state_dict(state); model.eval()

    app = FastAPI()

    @app.post("/generate")
    def generate(req: GenPayload):
        ids = tok(req.text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids)
        last = logits[0, -1]
        topk = min(max(req.top_k, 1), 20)
        idx = int(last.topk(topk).indices[0])
        return {"next_token": tok.decode([idx])}

    import uvicorn
    uvicorn.run(app, host=host, port=port)

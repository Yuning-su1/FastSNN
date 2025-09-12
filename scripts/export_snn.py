import torch, json, os, yaml
from fastsnn.builder.model_from_config import TinySNNLM
from transformers import AutoTokenizer

ckpt = torch.load("checkpoints/fastsnn_tiny.pt", map_location="cpu")
cfg = ckpt["cfg"]; os.makedirs("artifacts", exist_ok=True)

# 存 weights
torch.save(ckpt["model"], "artifacts/weights.safetensors")  # 简化：用 pt 也行

# 存 config / meta
with open("artifacts/config.json","w") as f:
    json.dump(cfg["model"], f, indent=2)
with open("artifacts/meta.json","w") as f:
    json.dump({"tokenizer": cfg["data"]["tokenizer"]}, f, indent=2)

print("Exported .snn-like bundle to artifacts/")

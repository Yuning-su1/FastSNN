from fastapi import FastAPI
import torch, json
from transformers import AutoTokenizer
from fastsnn.builder.model_from_config import TinySNNLM
import uvicorn

app = FastAPI()
tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
cfg = json.load(open("artifacts/config.json"))
state = torch.load("artifacts/weights.safetensors", map_location="cpu")
model = TinySNNLM(cfg, vocab_size=tok.vocab_size); model.load_state_dict(state); model.eval()

@app.post("/generate")
def generate(payload: dict):
    text = payload.get("text","")
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids)
    # 简化：只回最后一步 top-k 采样
    last = logits[0, -1]
    next_id = int(torch.topk(last, k=5).indices[0])
    return {"next_token": tok.decode([next_id])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

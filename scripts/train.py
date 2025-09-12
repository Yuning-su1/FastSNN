import os, yaml, math, torch, random
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from fastsnn.builder.model_from_config import TinySNNLM
from tqdm import tqdm

def set_seed(sd=42):
    random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def batch_iter(ds, tokenizer, seq_len, micro_bsz):
    buf, ids = "", []
    for ex in ds:
        buf += ex["text"] + "\n"
        if len(buf) > 20000:
            toks = tokenizer(buf, return_tensors=None)["input_ids"]
            ids.extend(toks); buf = ""
            while len(ids) >= seq_len*micro_bsz:
                mb = []
                for _ in range(micro_bsz):
                    mb.append(ids[:seq_len]); ids = ids[seq_len:]
                yield torch.tensor(mb, dtype=torch.long)

def main(cfg_path="configs/tiny.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["runtime"]["seed"])
    device = "cuda" if torch.cuda.is_available() and cfg["runtime"]["device"]!="cpu" else "cpu"
    dtype  = torch.bfloat16 if (device=="cuda" and cfg["runtime"]["dtype"]=="bf16") else torch.float32

    tok = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"])
    tok.pad_token = tok.eos_token

    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["subset"], streaming=cfg["data"]["streaming"])["train"]
    model = TinySNNLM(cfg["model"], vocab_size=tok.vocab_size).to(device).to(dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched_warm = cfg["train"]["warmup_steps"]; max_steps = cfg["train"]["max_steps"]

    step = 0; model.train()
    pbar = tqdm(batch_iter(ds, tok, cfg["train"]["seq_len"], cfg["train"]["micro_batch_size"]))
    for x in pbar:
        x = x.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda" and dtype==torch.bfloat16)):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                                     x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()

        # 线性 warmup + 余弦
        if step < sched_warm:
            for pg in opt.param_groups:
                pg["lr"] = cfg["train"]["lr"] * (step+1)/sched_warm
        else:
            progress = (step - sched_warm) / max(1, max_steps - sched_warm)
            for pg in opt.param_groups:
                pg["lr"] = 0.5 * cfg["train"]["lr"] * (1 + math.cos(math.pi*progress))

        if step % cfg["train"]["eval_interval"] == 0:
            pbar.set_description(f"step {step} loss {loss.item():.3f}")

        step += 1
        if step >= max_steps:
            break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, "checkpoints/fastsnn_tiny.pt")
    print("Saved to checkpoints/fastsnn_tiny.pt")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/tiny.yaml")

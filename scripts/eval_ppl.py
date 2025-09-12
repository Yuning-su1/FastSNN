import torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from fastsnn.builder.model_from_config import TinySNNLM
import math

@torch.no_grad()
def ppl(model, tok, ds, seq_len=2048, device="cpu"):
    model.eval()
    nll_sum, tok_cnt = 0.0, 0
    buf = ""
    for ex in ds.take(2000):  # 取一小段即可
        buf += ex["text"] + "\n"
        if len(buf) > 20000:
            ids = tok(buf)["input_ids"]; buf = ""
            for i in range(0, len(ids)-seq_len-1, seq_len):
                x = torch.tensor([ids[i:i+seq_len]], device=device)
                y = torch.tensor([ids[i+1:i+seq_len+1]], device=device)
                logits = model(x)
                nll = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                                        y.reshape(-1), reduction="sum")
                nll_sum += nll.item()
                tok_cnt += (seq_len)
    return math.exp(nll_sum / max(tok_cnt, 1))

if __name__ == "__main__":
    import sys
    cfg = yaml.safe_load(open(sys.argv[1]))
    tok = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"]); tok.pad_token = tok.eos_token
    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["subset"], streaming=True)["validation"]
    ckpt = torch.load("checkpoints/fastsnn_tiny.pt", map_location="cpu")
    model = TinySNNLM(ckpt["cfg"]["model"], vocab_size=tok.vocab_size)
    model.load_state_dict(ckpt["model"])
    print("PPL:", ppl(model, tok, ds))

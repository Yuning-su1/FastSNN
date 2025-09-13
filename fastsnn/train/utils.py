import torch
from contextlib import contextmanager

def setup_dtype(device: str, prefer_bf16: bool):
    if device == "cuda" and prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32

@contextmanager
def autocast_context(device: str, dtype):
    if device == "cuda" and dtype == torch.bfloat16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        # CPU/FP32 等场景不启用 autocast
        yield

def clip_gradients(model, max_norm: float):
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

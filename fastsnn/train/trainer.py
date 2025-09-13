import math, os, itertools
from typing import Callable, Optional, Dict, Any
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer

from .schedulers import WarmupCosine
from .utils import setup_dtype, autocast_context, clip_gradients

class Trainer:
    """
    Minimal yet practical training loop for FastSNN:
      - streaming HF datasets
      - CE loss
      - Warmup+Cosine LR schedule
      - AMP (bf16) on CUDA, FP32 on CPU
      - Grad clip
      - Hooks: log/eval/checkpoint
    """
    def __init__(
        self,
        model: nn.Module,
        dataset: str = "wikitext/wikitext-2-raw-v1",
        text_column: str = "text",
        tokenizer: str = "gpt2",
        seq_len: int = 2048,
        micro_batch_size: int = 2,
        global_batch_size: int = 8,
        base_lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_steps: int = 1000,
        warmup_steps: int = 200,
        grad_clip: float = 1.0,
        device: str = "auto",
        prefer_bf16: bool = True,
        seed: int = 42,
        streaming: bool = True,
        hooks: Optional[Dict[str, Callable]] = None,
    ):
        torch.manual_seed(seed)
        self.model = model
        self.dataset = dataset
        self.text_column = text_column
        self.tokenizer_name = tokenizer
        self.seq_len = seq_len
        self.micro_bsz = micro_batch_size
        self.global_bsz = global_batch_size
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.streaming = streaming

        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.dtype = setup_dtype(self.device, prefer_bf16)

        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.ds = load_dataset(*self.dataset.split("/"), streaming=self.streaming)["train"]
        self.val = load_dataset(*self.dataset.split("/"), streaming=True)["validation"]

        self._build_optimizer()
        self.set_scheduler(base_lr=self.base_lr, warmup_steps=self.warmup_steps, max_steps=self.max_steps)

        self.model.to(self.device).to(self.dtype)
        self.model.train()

        self.hooks = hooks or {}

        # streaming iterator
        self._iter = None

    # ------------- public API -------------
    def fit(self):
        step = 0
        while step < self.max_steps:
            batch = self._next_batch()
            if batch is None:
                self._rebuild_data()
                continue

            x = batch.to(self.device)
            with autocast_context(self.device, self.dtype):
                logits = self.model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1)
                )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(self.model, self.grad_clip)
            self.opt.step()

            lr = self.sched.step(self.opt, step)

            # hooks
            if "log" in self.hooks:
                self.hooks["log"](step, float(loss.item()), lr)
            if "eval" in self.hooks:
                self.hooks["eval"](step, self.model)
            if "checkpoint" in self.hooks:
                self.hooks["checkpoint"](step, self.model, extra_state={"cfg": self.export_state()})

            step += 1

        return self

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.export_state()}, path)
        return path

    def export_state(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "tokenizer": self.tokenizer_name,
            "seq_len": self.seq_len,
            "micro_batch_size": self.micro_bsz,
            "global_batch_size": self.global_bsz,
            "base_lr": self.base_lr,
            "weight_decay": self.weight_decay,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "grad_clip": self.grad_clip,
            "device": self.device,
            "dtype": str(self.dtype),
        }

    # ------------- internals -------------
    def _build_optimizer(self):
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)

    def set_scheduler(self, base_lr, warmup_steps, max_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.sched = WarmupCosine(base_lr=base_lr, warmup_steps=warmup_steps, max_steps=max_steps)

    def _reset_optimizer(self):
        self._build_optimizer()
        self.set_scheduler(self.base_lr, self.warmup_steps, self.max_steps)

    def _rebuild_data(self):
        self._iter = self._data_iter(self.ds, self.tok, self.seq_len, self.micro_bsz)

    def _next_batch(self):
        if self._iter is None:
            self._rebuild_data()
        try:
            return next(self._iter)
        except StopIteration:
            return None

    # streaming text → tokens → batches
    @staticmethod
    def _data_iter(ds, tok, seq_len, micro_bsz):
        buf = ""
        ids = []
        for ex in ds:
            buf += ex["text"] + "\n"
            if len(buf) > 20000:
                toks = tok(buf, return_tensors=None)["input_ids"]
                ids.extend(toks); buf = ""
                while len(ids) >= seq_len * micro_bsz:
                    mb = []
                    for _ in range(micro_bsz):
                        mb.append(ids[:seq_len]); ids = ids[seq_len:]
                    yield torch.tensor(mb, dtype=torch.long)

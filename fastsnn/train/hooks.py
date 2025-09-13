import time
from dataclasses import dataclass

@dataclass
class LogHook:
    log_every: int = 50
    def __call__(self, step, loss, lr, extra=None):
        if step % self.log_every == 0:
            msg = f"[step {step}] loss={loss:.4f} lr={lr:.2e}"
            if extra:
                msg += " " + " ".join([f"{k}={v}" for k,v in extra.items()])
            print(msg)

@dataclass
class EvalHook:
    eval_every: int = 200
    evaluator=None   # func(model) -> dict
    def __call__(self, step, model):
        if (self.evaluator is not None) and (step % self.eval_every == 0) and step > 0:
            model.eval()
            with torch.no_grad():
                metrics = self.evaluator(model)
            model.train()
            print("[eval]", " ".join([f"{k}={v}" for k,v in metrics.items()]))

@dataclass
class CheckpointHook:
    save_every: int = 500
    saver=None  # func(step, state_dict) -> path
    def __call__(self, step, model, extra_state=None):
        if (self.saver is not None) and (step % self.save_every == 0) and step > 0:
            to_save = {"model": model.state_dict()}
            if extra_state:
                to_save.update(extra_state)
            path = self.saver(step, to_save)
            print(f"[ckpt] saved → {path}")

# 依赖延迟导入：允许无 torch 环境解析本文件
import torch  # noqa: E402

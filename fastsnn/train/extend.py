"""
Context-length extension during conversion: re-init dataloader/scheduler and do a small LR warmup.
"""
from dataclasses import dataclass

@dataclass
class ExtendConfig:
    new_seq_len: int
    warmup_ratio: float = 0.2  # relative to previous max_steps
    resume_optimizer: bool = True

def extend_context(model, trainer, cfg: ExtendConfig):
    """
    - Rebuild dataloader with new seq_len
    - Do small LR warmup (e.g., 20% of previous max_steps)
    - Keep optimizer state to preserve stability (optional)
    """
    # 1) 更新数据流水
    trainer.seq_len = int(cfg.new_seq_len)
    trainer._rebuild_data()  # 重新构建数据迭代器（内部会使用新的 seq_len）

    # 2) 小学习率 warmup
    warm_steps = max(1, int(trainer.max_steps * cfg.warmup_ratio))
    old_base_lr = trainer.base_lr
    small_base_lr = old_base_lr * 0.2  # 以较小 LR 重新预热
    trainer.set_scheduler(base_lr=small_base_lr, warmup_steps=warm_steps, max_steps=trainer.max_steps)

    # 3) 是否保留优化器状态
    if not cfg.resume_optimizer:
        trainer._reset_optimizer()

    print(f"[extend] seq_len → {trainer.seq_len}, warmup_steps={warm_steps}, base_lr={small_base_lr:.2e}")
    return trainer


import torch
from fastsnn.builder.config import SNNConfig
from fastsnn.builder.model_from_config import build_model_from_config
from fastsnn.train.trainer import Trainer, TrainConfig

def synthetic_data(num_batches=20, B=2, T=64, V=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(num_batches):
        yield torch.randint(0, V, (B, T), generator=g)

if __name__ == '__main__':
    cfg = SNNConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2, d_ff=256, attn_kind='hybrid_alt')
    model = build_model_from_config(cfg)
    trainer = Trainer(model, TrainConfig(max_steps=10, log_every=2, lr=1e-3))
    trainer.fit(synthetic_data())
    print('train OK')

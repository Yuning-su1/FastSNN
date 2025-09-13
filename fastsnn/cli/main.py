
import argparse, torch
from fastsnn.builder.config import SNNConfig
from fastsnn.builder.model_from_config import build_model_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab_size', type=int, default=2000)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--attn', type=str, default='linear')
    args = ap.parse_args()
    cfg = SNNConfig(vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, attn_kind=args.attn)
    model = build_model_from_config(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = model(x)
    print('OK forward', tuple(y.shape))

if __name__ == '__main__':
    main()

import typer, yaml
from typing import Optional
import torch

from fastsnn.builder.config import SNNConfig, save_config, load_config
from fastsnn.builder.model_from_config import build_model_from_config
from fastsnn.deploy import export_snn, serve_snn

app = typer.Typer(help="FastSNN command-line interface")

@app.command()
def build(config: Optional[str] = typer.Option(None, help="YAML/JSON config"),
          out: str = typer.Option("checkpoints/build.pt", help="Path to save .pt"),
          d_model: int = 256, n_layers: int = 4, n_heads: int = 4, d_ff: int = 1024,
          attn: str = "hybrid_alt", phi: str = "relu",
          neuron: str = "adaptive_threshold", tau: float = 0.5, theta: float = 1.0, ste_tau: float = 1.0,
          dropout: float = 0.0, vocab_size: int = 50257):
    """
    Build a model and save a state_dict.
    """
    if config:
        cfg = load_config(config)
    else:
        cfg = SNNConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                        attn_kind=attn, attn_phi=phi,
                        neuron_type=neuron, neuron_tau=tau, neuron_theta=theta, neuron_ste_tau=ste_tau,
                        dropout_p=dropout)
    model = build_model_from_config(cfg)
    typer.echo(f"Built model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, out)
    typer.echo(f"Saved to {out}")

@app.command()
def export(checkpoint: str = typer.Argument(..., help="Path to .pt saved by build/train"),
           out_dir: str = typer.Option("artifacts/model.snn", help="Export bundle dir")):
    """
    Export a .snn bundle from a checkpoint.
    """
    blob = torch.load(checkpoint, map_location="cpu")
    cfg = SNNConfig(**blob.get("cfg", {}))
    from fastsnn.builder.model_from_config import build_model_from_config
    model = build_model_from_config(cfg)
    model.load_state_dict(blob["state_dict"])
    export_snn(model, cfg.__dict__, out_dir)
    typer.echo(f"Exported to {out_dir}")

@app.command()
def serve(bundle: str = typer.Argument(..., help="Path to .snn bundle directory"),
          host: str = "0.0.0.0", port: int = 8000):
    serve_snn(bundle, host=host, port=port)

if __name__ == "__main__":
    app()

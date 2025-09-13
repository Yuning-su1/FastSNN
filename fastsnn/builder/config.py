
from dataclasses import dataclass, asdict
from typing import Optional
import json, yaml

@dataclass
class SNNConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    attn_kind: str = 'linear'   # 'linear'|'sliding'|'hybrid_alt'
    window: int = 128
    dropout: float = 0.0
    tie_lm_head: bool = True

    def save(self, path: str):
        obj = asdict(self)
        if path.endswith('.json'):
            json.dump(obj, open(path,'w'), indent=2)
        else:
            yaml.safe_dump(obj, open(path,'w'))

    @classmethod
    def load(cls, path: str) -> 'SNNConfig':
        if path.endswith('.json'):
            data = json.load(open(path,'r'))
        else:
            data = yaml.safe_load(open(path,'r'))
        return cls(**data)

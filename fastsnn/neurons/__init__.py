from ..core.registry import Registry

NeuronRegistry = Registry("NeuronRegistry")

# 导出基础类与已实现神经元
from .surrogate import SurrogateSTE                      # noqa: F401
from .lif_sint import LIFsINT                            # noqa: F401
from .adex_sint import AdExsINT                          # noqa: F401
from .adaptive_threshold import AdaptiveThresholdNeuron  # noqa: F401

# 正确的注册表实现
NeuronRegistry.table.update({
    "lif_sint": LIFsINT,
    "adex_sint": AdExsINT,
    "adaptive_threshold": AdaptiveThresholdNeuron,
})

def build_neuron(kind: str, d_model: int, **kwargs):
    kind = (kind or "lif_sint").lower()
    if kind not in NeuronRegistry.table:
        raise KeyError(f"Unknown neuron type: {kind}. Available: {list(NeuronRegistry.table.keys())}")
    return NeuronRegistry.table[kind](d_model=d_model, **kwargs)

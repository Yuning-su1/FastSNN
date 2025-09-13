
from ..core.registry import Registry
from .lif_sint import LIFsINT
from .adaptive_threshold import AdaptiveThresholdNeuron

NeuronRegistry = Registry("NeuronRegistry")
NeuronRegistry.update({
    'lif_sint': LIFsINT,
    'adaptive_threshold': AdaptiveThresholdNeuron,
})

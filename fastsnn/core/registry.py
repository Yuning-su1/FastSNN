class Registry:
    """
    Simple registry for pluggable modules.
    Example:
        NEURON_REG = Registry("neuron")
        @NEURON_REG.register("lif_sint")
        class LIFsINT(...): ...
    """
    def __init__(self, name):
        self.name = name
        self.table = {}

    def register(self, key):
        def decorator(obj):
            if key in self.table:
                raise KeyError(f"{key} already registered in {self.name}")
            self.table[key] = obj
            return obj
        return decorator

    def get(self, key):
        if key not in self.table:
            raise KeyError(f"{key} not found in {self.name}")
        return self.table[key]

    def list_keys(self):
        return list(self.table.keys())

    def __repr__(self):
        return f"Registry<{self.name}>({list(self.table.keys())})"

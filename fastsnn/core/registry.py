
class Registry:
    """Simple registry for pluggable components."""
    def __init__(self, name: str):
        self.name = name
        self.table = {}
    def register(self, key: str):
        def deco(obj):
            if key in self.table:
                raise KeyError(f"{key} already in {self.name}")
            self.table[key] = obj
            return obj
        return deco
    def get(self, key: str):
        if key not in self.table:
            raise KeyError(f"{key} not found in {self.name}")
        return self.table[key]
    def update(self, mapping: dict):
        for k,v in mapping.items():
            self.table[k] = v
    def list_keys(self):
        return list(self.table.keys())
    def __repr__(self):
        return f"Registry<{self.name}>({list(self.table.keys())})"

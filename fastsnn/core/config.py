import json, yaml

class SNNConfig(dict):
    """
    Minimal config wrapper for SNN models.
    Behaves like a dict but supports load/save to JSON/YAML.
    """
    @classmethod
    def load(cls, path):
        if path.endswith(".json"):
            return cls(json.load(open(path)))
        elif path.endswith(".yaml") or path.endswith(".yml"):
            return cls(yaml.safe_load(open(path)))
        else:
            raise ValueError("Unsupported config format")

    def save(self, path):
        if path.endswith(".json"):
            json.dump(self, open(path,"w"), indent=2)
        elif path.endswith(".yaml") or path.endswith(".yml"):
            yaml.safe_dump(dict(self), open(path,"w"))
        else:
            raise ValueError("Unsupported config format")

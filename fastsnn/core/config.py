
import json, yaml

class Config(dict):
    @classmethod
    def load(cls, path: str):
        if path.endswith(('.yml','.yaml')):
            return cls(yaml.safe_load(open(path, 'r', encoding='utf-8')))
        elif path.endswith('.json'):
            return cls(json.load(open(path, 'r', encoding='utf-8')))
        raise ValueError('Unsupported config file')
    def save(self, path: str):
        if path.endswith(('.yml','.yaml')):
            yaml.safe_dump(dict(self), open(path, 'w', encoding='utf-8'))
        elif path.endswith('.json'):
            json.dump(dict(self), open(path, 'w', encoding='utf-8'), indent=2)
        else:
            raise ValueError('Unsupported config file')

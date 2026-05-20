class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, type=None):
        def wrapper(cls):
            key = type or cls.__name__
            self._registry[key] = cls
            return cls
        return wrapper

    def get(self, type):
        if type not in self._registry:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(f"Unknown model type '{type}'. Available models: {available}")
        return self._registry[type]

    def build(self, cfg: dict):
        if cfg is None or cfg.get('type') is None:
            return None
        cls = self.get(cfg['type'])
        return cls(**{k: v for k, v in cfg.items() if k != 'type'})

MODELS = Registry()

def build_net(cfg):
    """Build head."""
    return MODELS.build(cfg)







class bidict(dict):
    def __init__(self, *args, **kwargs):
        self.direct = dict(*args, **kwargs)
        self.inverse = {v: k for k, v in self.direct.items()}
        super().__init__({**self.direct, **self.inverse})

    def __setitem__(self, key, value):
        self.direct[key] = value
        self.inverse[value] = key
        super().__setitem__(key, value)
        super().__setitem__(value, key)

    def __delitem__(self, key):
        del self.inverse[self[key]]
        del self.direct[key]
        super().__delitem__(key)
        super().__delitem__(self[key])

    def __repr__(self):
        return f'{super().__repr__()}'

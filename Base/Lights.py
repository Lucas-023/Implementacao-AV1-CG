import torch


class Light:
    def __init__(self):
        pass

    def position(self):
        raise NotImplementedError("Subclasses should implement this method")


class PointLight:
    def __init__(self, pos, color, intensity):
        self.pos = pos
        self.color = color
        self.intensity = intensity

    def position(self):
        return self.pos


class AreaLight:
    pass

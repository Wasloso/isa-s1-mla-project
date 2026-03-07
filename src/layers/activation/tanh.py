from src.layers.base import Layer
from src.types import Array


class Tanh(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True
        self.output = None

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: Array, training: bool = True) -> Array:
        self.output = self.xp.tanh(x)
        return self.output

    def backward(self, output_gradient: Array) -> Array:
        assert self.output is not None
        return output_gradient * (1.0 - self.output**2)

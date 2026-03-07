from src.layers.base import Layer
from src.types import Array


class ReLU(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: Array, training: bool = True) -> Array:
        self.input = x
        return self.xp.maximum(0, x)

    def backward(self, output_gradient: Array) -> Array:
        assert self.input is not None
        relu_gradient = (self.input > 0).astype(float)
        return output_gradient * relu_gradient

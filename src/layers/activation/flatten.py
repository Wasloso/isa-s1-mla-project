from src.layers.base import Layer
from src.types import Array


class Flatten(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.input_shape = None
        self.built = True

    def build(self, input_shape) -> None:
        pass

    def forward(self, x: Array, training: bool = True, **kwargs) -> Array:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, output_gradient: Array) -> Array:
        return output_gradient.reshape(self.input_shape)

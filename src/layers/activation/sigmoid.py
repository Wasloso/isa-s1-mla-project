from src.layers.base import Layer
from src.types import Array


class Sigmoid(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: Array, training: bool = True, **kwargs) -> Array:
        self.output = 1 / (1 + self.xp.exp(-x))
        return self.output

    def backward(self, output_gradient: Array) -> Array:
        sigmoid_grad = self.output * (1 - self.output)
        return output_gradient * sigmoid_grad

from src.layers.base import Layer
from src.types import Array


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01, name: str | None = None):
        super().__init__(name=name)
        self.alpha = alpha
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: Array, training: bool = True, **kwargs) -> Array:
        self.input = x
        return self.xp.where(x > 0, x, self.alpha * x)

    def backward(self, output_gradient: Array) -> Array:
        return output_gradient * self.xp.where(self.input > 0, 1.0, self.alpha)

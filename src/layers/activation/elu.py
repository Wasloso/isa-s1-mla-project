from src.layers.base import Layer
from src.types import Array


class ELU(Layer):
    def __init__(self, alpha: float = 1.0, name: str | None = None):
        super().__init__(name=name)
        self.alpha = alpha
        self.built = True
        self.input = None

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: Array, training: bool = True) -> Array:
        self.input = x
        return self.xp.where(x > 0, x, self.alpha * (self.xp.exp(x) - 1.0))

    def backward(self, output_gradient: Array) -> Array:
        assert self.input is not None
        return output_gradient * self.xp.where(self.input > 0, 1.0, self.alpha * self.xp.exp(self.input))

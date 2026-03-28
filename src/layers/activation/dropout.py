from src.layers.base import Layer
from src.types import Array


class Dropout(Layer):
    def __init__(self, rate: float = 0.5, name: str | None = None):
        super().__init__(name=name)
        self.rate = rate
        self.mask = None
        self.built = True

    def build(self, input_shape: tuple) -> None:
        self.input_shape = input_shape

    def forward(self, x: Array, training: bool = True, **kwargs) -> Array:
        if not training or self.rate == 0:
            return x
        self.mask = self.xp.random.rand(*x.shape) > self.rate
        scale = self.xp.float32(1.0 / (1.0 - self.rate))
        return x * self.mask * scale

    def backward(self, output_gradient: Array) -> Array:
        scale = self.xp.float32(1.0 / (1.0 - self.rate))
        return output_gradient * self.mask * scale

from src.layers.base import Layer
from src.types import Array


class Softmax(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: Array, training: bool = True, **kwargs) -> Array:
        exps = self.xp.exp(x - self.xp.max(x, axis=1, keepdims=True))
        self.output = exps / self.xp.sum(exps, axis=1, keepdims=True, dtype=self.xp.float32)
        return self.output

    def backward(self, output_gradient: Array) -> Array:
        return self.output * (
            output_gradient - self.xp.sum(output_gradient * self.output, axis=1, keepdims=True, dtype=self.xp.float32)
        )

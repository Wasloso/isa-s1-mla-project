from src.types import Array

from .base import Layer


class Dense(Layer):
    def __init__(self, units: int, name: str | None = None):
        super().__init__(name=name)
        self.units: int = units
        self.kernel: Array | None = None
        self.bias: Array | None = None

    def build(self, input_shape: tuple) -> None:
        input_dim = input_shape[-1]
        std = self.xp.sqrt(2.0 / input_dim)
        self.kernel = self.xp.random.randn(input_dim, self.units) * std
        self.bias = self.xp.zeros((1, self.units))
        self.trainable_weights = [self.kernel, self.bias]  # type: ignore
        self.gradients = [self.xp.zeros_like(self.kernel), self.xp.zeros_like(self.bias)]
        self.built = True

    def forward(self, x: Array, training: bool = True) -> Array:
        self.input = x
        return self.xp.dot(x, self.kernel) + self.bias  # type: ignore

    def backward(self, output_gradient: Array) -> Array:
        self.gradients[0] = self.xp.dot(self.input.T, output_gradient)
        self.gradients[1] = self.xp.sum(output_gradient, axis=0, keepdims=True)
        return self.xp.dot(output_gradient, self.kernel.T)  # type: ignore

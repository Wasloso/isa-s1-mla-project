import numpy as np

from .base import Layer


class Dense(Layer):
    def __init__(self, units: int, name: str | None = None):
        super().__init__(name=name)
        self.units: int = units
        self.kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def build(self, input_shape: tuple) -> None:
        input_dim = input_shape[-1]
        std = np.sqrt(2.0 / input_dim)
        self.kernel = np.random.randn(input_dim, self.units) * std
        self.bias = np.zeros((1, self.units))
        self.trainable_weights = [self.kernel, self.bias]  # type: ignore
        self.gradients = [np.zeros_like(self.kernel), np.zeros_like(self.bias)]
        self.built = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        return np.dot(x, self.kernel) + self.bias  # type: ignore

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        self.gradients[0] = np.dot(self.input.T, output_gradient)
        self.gradients[1] = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.kernel.T)  # type: ignore

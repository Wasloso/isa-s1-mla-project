import numpy as np

from .base import Layer


class ReLU(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.input is not None
        relu_gradient = (self.input > 0).astype(float)
        return output_gradient * relu_gradient

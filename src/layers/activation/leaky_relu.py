import numpy as np

from src.layers.base import Layer


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01, name: str | None = None):
        super().__init__(name=name)
        self.alpha = alpha
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * np.where(self.input > 0, 1.0, self.alpha)

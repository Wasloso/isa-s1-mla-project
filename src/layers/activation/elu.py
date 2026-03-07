import numpy as np

from src.layers.base import Layer


class ELU(Layer):
    def __init__(self, alpha: float = 1.0, name: str | None = None):
        super().__init__(name=name)
        self.alpha = alpha
        self.built = True
        self.input = None

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.input is not None
        return output_gradient * np.where(self.input > 0, 1.0, self.alpha * np.exp(self.input))

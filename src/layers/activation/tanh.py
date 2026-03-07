import numpy as np

from src.layers.base import Layer


class Tanh(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True
        self.output = None

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.output is not None
        return output_gradient * (1.0 - self.output**2)

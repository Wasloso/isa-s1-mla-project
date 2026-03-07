import numpy as np

from src.layers.base import Layer


class Sigmoid(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        sigmoid_grad = self.output * (1 - self.output)
        return output_gradient * sigmoid_grad

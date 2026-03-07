import numpy as np

from src.layers.base import Layer


class Softmax(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.built = True

    def build(self, input_shape: tuple):
        pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # TODO: Implement proper softmax backward pass
        return output_gradient

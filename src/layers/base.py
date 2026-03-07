from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, name: str | None = None):
        self.name = name
        self.built = False
        self.input = None
        self.trainable_weights: list[np.ndarray] = []
        self.gradients: list[np.ndarray] = []

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.built:
            self.build(x.shape)
            self.built = True
        return self.forward(x, training)

    @abstractmethod
    def build(self, input_shape) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        pass

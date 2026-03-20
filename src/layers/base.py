from abc import ABC, abstractmethod

from src.backend import backend
from src.types import Array


class Layer(ABC):
    def __init__(self, name: str | None = None):
        self.name = name
        self.built = False
        self.input = None
        self.trainable_weights: list[Array] = []
        self.gradients: list[Array] = []

    def __call__(self, x: Array, training: bool = True) -> Array:
        if not self.built:
            self.build(x.shape)
            self.built = True
        return self.forward(x, training)

    @abstractmethod
    def build(self, input_shape) -> None:
        pass

    @abstractmethod
    def forward(self, x: Array, training: bool = True) -> Array:
        pass

    @abstractmethod
    def backward(self, output_gradient: Array) -> Array:
        pass

    @property
    def xp(self):
        return backend.xp

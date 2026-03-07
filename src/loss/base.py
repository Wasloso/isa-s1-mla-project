from abc import ABC, abstractmethod

from src.backend import backend
from src.types import Array


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: Array, y_true: Array) -> Array:
        pass

    @abstractmethod
    def backward(self, y_pred: Array, y_true: Array) -> Array:
        pass

    @property
    def xp(self):
        return backend.xp

from abc import ABC, abstractmethod

from src.backend import backend
from src.layers.base import Layer


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layers: list[Layer]) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def xp(self):
        return backend.xp

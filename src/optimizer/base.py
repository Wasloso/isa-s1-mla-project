from abc import ABC, abstractmethod

from src.layers.base import Layer


class Optimizer(ABC):
    @abstractmethod
    def update(self, layers: list[Layer]) -> None:
        pass

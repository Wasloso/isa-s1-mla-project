from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

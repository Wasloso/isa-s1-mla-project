from abc import ABC, abstractmethod

import numpy as np

from src.loss.base import Loss
from src.optimizer.base import Optimizer


class Network(ABC):
    def __init__(self) -> None:
        self.compiled = False

    @abstractmethod
    def predict(self, x: np.ndarray):
        pass

    @abstractmethod
    def train_step(self, x: np.ndarray, y: np.ndarray):
        pass

    def compile(self, loss: Loss, optimizer: Optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.compiled = True

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, verbose: int = 10):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training.")
        history = []

        for epoch in range(epochs):
            loss_val = self.train_step(x_train, y_train)
            history.append(loss_val)
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_val:.6f}")

        return history

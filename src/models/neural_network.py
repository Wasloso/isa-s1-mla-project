from abc import ABC, abstractmethod

from src.backend import backend
from src.loss.base import Loss
from src.optimizer.base import Optimizer
from src.types import Array


class Network(ABC):
    def __init__(self) -> None:
        self.compiled = False

    def predict(self, x: Array):
        x_dev = backend.asarray(x)
        pred_dev = self._predict(x_dev)
        return backend.to_numpy(pred_dev)

    @abstractmethod
    def _predict(self, x: Array):
        pass

    @abstractmethod
    def train_step(self, x: Array, y: Array):
        pass

    def compile(self, loss: Loss, optimizer: Optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.compiled = True

    def fit(self, x_train: Array, y_train: Array, epochs: int, verbose: int = 10):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training.")
        history = []
        x_dev = backend.asarray(x_train)
        y_dev = backend.asarray(y_train)

        for epoch in range(epochs):
            loss_val = self.train_step(x_dev, y_dev)
            history.append(backend.to_numpy(loss_val))
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_val:.6f}")

        return history

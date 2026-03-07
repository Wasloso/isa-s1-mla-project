from src.types import Array

from .base import Loss


class MSE(Loss):
    def forward(self, y_pred: Array, y_true: Array):
        return self.xp.mean(self.xp.power(y_true - y_pred, 2))

    def backward(self, y_pred: Array, y_true: Array):
        return 2 * (y_pred - y_true) / y_true.size

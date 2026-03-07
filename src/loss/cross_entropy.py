from src.types import Array

from .base import Loss


class CrossEntropy(Loss):
    def forward(self, y_pred: Array, y_true: Array) -> Array:
        y_pred_clipped = self.xp.clip(y_pred, 1e-7, 1 - 1e-7)
        return -self.xp.mean(self.xp.sum(y_true * self.xp.log(y_pred_clipped), axis=1))

    def backward(self, y_pred: Array, y_true: Array) -> Array:
        return (y_pred - y_true) / y_true.shape[0]

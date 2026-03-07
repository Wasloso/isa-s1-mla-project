import numpy as np

from src.layers.base import Layer
from src.models.neural_network import Network


class Sequential(Network):
    def __init__(self, layers: list[Layer] | None = None):
        super().__init__()
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def train_step(self, x: np.ndarray, y: np.ndarray):
        assert self.loss is not None and self.optimizer is not None
        pred = self.predict(x)
        loss_val = self.loss.forward(pred, y)

        grad = self.loss.backward(pred, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.optimizer.update(self.layers)

        return loss_val

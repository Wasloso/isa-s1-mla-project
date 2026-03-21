from src.layers.base import Layer
from src.optimizer.base import Optimizer


class SGD(Optimizer):
    def update(self, layers: list[Layer]) -> None:
        for layer in layers:
            if not hasattr(layer, "trainable_weights") or len(layer.trainable_weights) == 0:
                continue

            if not hasattr(layer, "gradients") or len(layer.gradients) == 0:
                continue

            for i, _ in enumerate(layer.trainable_weights):
                if i < len(layer.gradients):
                    layer.trainable_weights[i] -= self.learning_rate * layer.gradients[i]

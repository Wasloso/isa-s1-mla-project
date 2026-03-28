from src.layers.base import Layer
from src.layers.conv1d import Conv1D
from src.models.neural_network import Network
from src.types import Array


class Sequential(Network):
    def __init__(self, layers: list[Layer] | None = None):
        super().__init__()
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)

    def _predict(self, x, training: bool = False, mask: Array | None = None):
        output = x
        current_mask = mask

        for layer in self.layers:
            output = layer(output, training=training, mask=current_mask)
            if isinstance(layer, Conv1D) and current_mask is not None:
                current_mask = (current_mask + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

        return output

    def train_step(self, x: Array, y: Array, mask: Array | None = None, training: bool = True) -> tuple[Array, Array]:
        assert self.loss is not None and self.optimizer is not None
        pred = self._predict(x, training=training, mask=mask)
        loss_val = self.loss.forward(pred, y)

        grad = self.loss.backward(pred, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.optimizer.update(self.layers)

        return loss_val, pred

    def reset(self):
        super().reset()
        for layer in self.layers:
            layer.reset()

from src.layers.base import Layer
from src.optimizer.base import Optimizer
from src.types import Array


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: dict[tuple[int, int], Array] = {}
        self.v: dict[tuple[int, int], Array] = {}
        self.t = 0

    def update(self, layers: list[Layer]) -> None:
        self.t += 1

        for layer in layers:
            if not hasattr(layer, "trainable_weights") or len(layer.trainable_weights) == 0:
                continue
            if not hasattr(layer, "gradients") or len(layer.gradients) == 0:
                continue

            for i, param in enumerate(layer.trainable_weights):
                if i >= len(layer.gradients):
                    continue

                grad = layer.gradients[i]
                key = (id(layer), i)

                if key not in self.m:
                    self.m[key] = self.xp.zeros_like(param)
                    self.v[key] = self.xp.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad**2)

                m_hat = self.m[key] / (1.0 - self.beta1**self.t)
                v_hat = self.v[key] / (1.0 - self.beta2**self.t)

                param -= self.learning_rate * m_hat / (self.xp.sqrt(v_hat) + self.epsilon)  # noqa: PLW2901

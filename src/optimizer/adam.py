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

    def reset(self):
        self.m.clear()
        self.v.clear()
        self.t = 0

    def update(self, layers: list[Layer]) -> None:
        self.t += 1
        bias_corr1 = 1.0 - self.beta1**self.t
        bias_corr2 = 1.0 - self.beta2**self.t
        lr_t = self.learning_rate * (self.xp.sqrt(bias_corr2) / bias_corr1)
        for layer in layers:
            if not hasattr(layer, "trainable_weights") or len(layer.trainable_weights) == 0:
                continue
            if not hasattr(layer, "gradients") or len(layer.gradients) == 0:
                continue

            for i, (param, grad) in enumerate(zip(layer.trainable_weights, layer.gradients, strict=False)):
                if i >= len(layer.gradients):
                    continue
                key = (id(layer), i)

                if key not in self.m:
                    self.m[key] = self.xp.zeros_like(param)
                    self.v[key] = self.xp.zeros_like(param)

                self.m[key] *= self.beta1
                self.m[key] += (1.0 - self.beta1) * grad
                self.v[key] *= self.beta2
                self.v[key] += (1.0 - self.beta2) * (grad**2)

                param -= lr_t * self.m[key] / (self.xp.sqrt(self.v[key]) + self.epsilon)  # noqa: PLW2901

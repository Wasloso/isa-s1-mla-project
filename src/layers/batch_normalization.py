import numpy as np

from src.layers.base import Layer
from src.types import Array


class BatchNormalization(Layer):
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9, name: str | None = None):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.x_centered = None
        self.std_inv = None
        self.x_norm = None
        self.axes = None

    def build(self, input_shape: tuple) -> None:
        self.axes = tuple(range(len(input_shape) - 1))
        channels = input_shape[-1]
        self.gamma = self.xp.ones((channels,), dtype=self.xp.float32)
        self.beta = self.xp.zeros((channels,), dtype=self.xp.float32)
        self.trainable_weights = [self.gamma, self.beta]
        self.gradients = [self.xp.zeros_like(self.gamma), self.xp.zeros_like(self.beta)]
        self.running_mean = self.xp.zeros((channels,), dtype=self.xp.float32)
        self.running_var = self.xp.ones((channels,), dtype=self.xp.float32)

    def forward(self, x: Array, training: bool = True, **kwargs) -> Array:
        self.input = x

        if training:
            mean = self.xp.mean(x, axis=self.axes)
            var = self.xp.var(x, axis=self.axes)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.x_centered = x - mean
            self.std_inv = 1.0 / self.xp.sqrt(var + self.epsilon)
            self.x_norm = self.x_centered * self.std_inv
        else:
            self.x_centered = x - self.running_mean
            self.std_inv = 1.0 / self.xp.sqrt(self.running_var + self.epsilon)
            self.x_norm = self.x_centered * self.std_inv

        return self.gamma * self.x_norm + self.beta

    def backward(self, output_gradient: Array) -> Array:
        N = self.xp.float32(self.input.size / self.gamma.size)

        dgamma = self.xp.sum(output_gradient * self.x_norm, axis=self.axes, dtype=self.xp.float32)
        dbeta = self.xp.sum(output_gradient, axis=self.axes, dtype=self.xp.float32)

        self.gradients = [dgamma, dbeta]

        dx_norm = output_gradient * self.gamma
        dvar = self.xp.sum(dx_norm * self.x_centered * -0.5 * (self.std_inv**3), axis=self.axes, dtype=self.xp.float32)
        dmean = self.xp.sum(dx_norm * -self.std_inv, axis=self.axes, dtype=self.xp.float32) + dvar * self.xp.mean(
            -2.0 * self.x_centered, axis=self.axes, dtype=self.xp.float32
        )

        dx = (dx_norm * self.std_inv) + (dvar * 2.0 * self.x_centered / N) + (dmean / N)
        return dx.astype(self.xp.float32, copy=False)

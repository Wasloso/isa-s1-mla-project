from src.layers.base import Layer
from src.types import Array


class MaskedGlobalAveragePool1D(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.input_shape = None
        self.true_lengths = None

    def build(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.built = True

    def forward(self, x: Array, training: bool = True, mask: Array | None = None) -> Array:
        if mask is None:
            raise ValueError("MaskedGlobalAveragePool1D requires a mask/lengths array.")
        if mask.shape[0] != x.shape[0]:
            raise ValueError(f"Mask batch size ({mask.shape[0]}) must match input batch size ({x.shape[0]}).")
        self.input_shape = x.shape
        self.true_lengths = mask
        x_sum = self.xp.sum(x, axis=2)
        mask_counts = self.xp.maximum(mask, 1.0).reshape(-1, 1)
        return x_sum / mask_counts

    def backward(self, output_gradient: Array) -> Array:
        mask_counts = self.xp.maximum(self.true_lengths, 1.0).reshape(-1, 1, 1)
        grad_expanded = self.xp.expand_dims(output_gradient, axis=2)
        return self.xp.broadcast_to(grad_expanded, self.input_shape) / mask_counts

from src.layers.base import Layer
from src.types import Array


class GlobalConcatPool1D(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self.input_shape = None
        self.true_lengths = None
        self.arg_max = None

    def build(self, input_shape: tuple):
        self.input_shape = input_shape
        self.built = True

    def forward(self, x: Array, training: bool = True, mask: Array | None = None, **kwargs) -> Array:
        self.input_shape = x.shape
        self.true_lengths = mask
        x_sum = self.xp.sum(x, axis=2)
        counts = self.xp.maximum(mask, 1.0).reshape(-1, 1) if mask is not None else float(x.shape[2])
        avg_p = x_sum / counts
        if training:
            self.arg_max = self.xp.argmax(x, axis=2)
        max_p = self.xp.max(x, axis=2)

        return self.xp.concatenate([avg_p, max_p], axis=1, dtype=self.xp.float32)

    def backward(self, output_gradient: Array) -> Array:
        """
        output_gradient: (batch, filters * 2)
        """
        if self.input_shape is None:
            raise RuntimeError("GlobalConcatPool1D.backward called before forward.")
        if self.arg_max is None:
            raise RuntimeError("GlobalConcatPool1D.backward requires arg_max from training forward pass.")
        batch_size, filters, time_steps = self.input_shape
        grad_avg_part = output_gradient[:, :filters]
        grad_max_part = output_gradient[:, filters:]
        if self.true_lengths is not None:
            counts = self.xp.maximum(self.true_lengths, 1.0).reshape(-1, 1, 1)
        else:
            counts = float(time_steps)
        grad_avg = self.xp.expand_dims(grad_avg_part, axis=2)
        grad_avg = self.xp.broadcast_to(grad_avg, self.input_shape) / counts
        grad_avg.astype(self.xp.float32, copy=False)
        grad_max = self.xp.zeros(self.input_shape, dtype=self.xp.float32)
        batch_idx = self.xp.arange(batch_size)[:, None]
        filter_idx = self.xp.arange(
            filters,
        )[None, :]
        grad_max[batch_idx, filter_idx, self.arg_max] += grad_max_part
        return grad_avg + grad_max

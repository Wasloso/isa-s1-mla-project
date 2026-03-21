from src.layers.base import Layer
from src.types import Array


class Conv1D(Layer):
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, padding: int = 0, name: str | None = None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = None
        self.bias = None
        self.x_padded = None
        self.windows = None

    def build(self, input_shape) -> None:
        in_channels = input_shape[1]
        limit = self.xp.sqrt(2 / (in_channels * self.kernel_size))
        self.kernel = self.xp.random.randn(self.filters, in_channels, self.kernel_size) * limit
        self.bias = self.xp.zeros((self.filters, 1))

        self.trainable_weights = [self.kernel, self.bias]
        self.built = True

    def forward(self, x: Array, training: bool = True, mask: Array | None = None) -> Array:
        batch_size, in_channels, in_len = x.shape
        if self.padding > 0:
            self.x_padded = self.xp.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")
        else:
            self.x_padded = x
        out_len = (in_len + 2 * self.padding - self.kernel_size) // self.stride + 1
        s0, s1, s2 = self.x_padded.strides
        self.windows = self.xp.lib.stride_tricks.as_strided(
            self.x_padded,
            shape=(batch_size, out_len, in_channels, self.kernel_size),
            strides=(s0, s2 * self.stride, s1, s2),
        )
        output = self.xp.tensordot(self.windows, self.kernel, axes=((2, 3), (1, 2)))
        return output.transpose(0, 2, 1) + self.bias

    def backward(self, output_gradient: Array) -> Array:
        dbias = self.xp.sum(output_gradient, axis=(0, 2)).reshape(self.filters, 1)
        dkernel = self.xp.tensordot(output_gradient, self.windows, axes=((0, 2), (0, 1)))
        dx_padded = self.xp.zeros_like(self.x_padded)
        s0, s1, s2 = dx_padded.strides
        dx_windows = self.xp.lib.stride_tricks.as_strided(
            dx_padded, shape=self.windows.shape, strides=(s0, s2 * self.stride, s1, s2)
        )
        dx_windows += self.xp.tensordot(output_gradient.transpose(0, 2, 1), self.kernel, axes=(2, 0))
        self.gradients = [dkernel, dbias]
        if self.padding > 0:
            return dx_padded[:, :, self.padding : -self.padding]
        return dx_padded

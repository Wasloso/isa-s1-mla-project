import numpy as np

from src.layers.activation.relu import ReLU


class TestReLU:
    """Test suite for ReLU activation layer."""

    def test_forward_positive_values(self):
        """Test forward pass with positive values."""
        relu = ReLU()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = relu(x)
        np.testing.assert_array_equal(output, x)

    def test_forward_negative_values(self):
        """Test forward pass with negative values."""
        relu = ReLU()
        x = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        expected = np.zeros_like(x)
        output = relu(x)
        np.testing.assert_array_equal(output, expected)

    def test_forward_mixed_values(self):
        """Test forward pass with mixed positive and negative values."""
        relu = ReLU()
        x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        expected = np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]])
        output = relu(x)
        np.testing.assert_array_equal(output, expected)

    def test_forward_zero_values(self):
        """Test forward pass with zero values."""
        relu = ReLU()
        x = np.array([[0.0, 0.0], [0.0, 0.0]])
        output = relu(x)
        np.testing.assert_array_equal(output, x)

    def test_forward_stores_input(self):
        """Test that forward pass stores the input for backward pass."""
        relu = ReLU()
        x = np.array([[1.0, -2.0, 3.0]])
        relu(x)
        np.testing.assert_array_equal(relu.input, x)

    def test_backward_positive_gradient(self):
        """Test backward pass for positive values."""
        relu = ReLU()
        x = np.array([[1.0, 2.0, 3.0]])
        relu(x)
        output_grad = np.array([[0.5, 0.5, 0.5]])
        grad = relu.backward(output_grad)
        np.testing.assert_array_equal(grad, output_grad)

    def test_backward_negative_gradient(self):
        """Test backward pass for negative values."""
        relu = ReLU()
        x = np.array([[-1.0, -2.0, -3.0]])
        relu(x)
        output_grad = np.array([[0.5, 0.5, 0.5]])
        expected = np.array([[0.0, 0.0, 0.0]])
        grad = relu.backward(output_grad)
        np.testing.assert_array_equal(grad, expected)

    def test_backward_mixed_gradient(self):
        """Test backward pass with mixed input (positive and negative)."""
        relu = ReLU()
        x = np.array([[1.0, -2.0, 3.0, -4.0]])
        relu(x)
        output_grad = np.array([[1.0, 2.0, 3.0, 4.0]])
        expected = np.array([[1.0, 0.0, 3.0, 0.0]])
        grad = relu.backward(output_grad)
        np.testing.assert_array_equal(grad, expected)

    def test_backward_batch(self):
        """Test backward pass with batch input."""
        relu = ReLU()
        x = np.array([[1.0, -2.0], [3.0, -4.0], [-5.0, 6.0]])
        relu(x)
        output_grad = np.ones_like(x)
        expected = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        grad = relu.backward(output_grad)
        np.testing.assert_array_equal(grad, expected)

    def test_backward_zero_at_boundary(self):
        """Test backward pass at zero boundary."""
        relu = ReLU()
        x = np.array([[0.0, 0.0, 0.0]])
        relu(x)
        output_grad = np.array([[1.0, 1.0, 1.0]])
        grad = relu.backward(output_grad)
        # At zero, gradient should be 0 (treated as negative)
        np.testing.assert_array_equal(grad, np.array([[0.0, 0.0, 0.0]]))

    def test_forward_large_values(self):
        """Test forward pass with large values."""
        relu = ReLU()
        x = np.array([[1e6, 1e10], [-1e6, -1e10]])
        output = relu(x)
        expected = np.array([[1e6, 1e10], [0.0, 0.0]])
        np.testing.assert_array_equal(output, expected)

    def test_gradient_flow_chain_rule(self):
        """Test that gradient flows correctly through chain rule."""
        relu = ReLU()
        x = np.array([[2.0, -1.0, 3.0]])
        relu(x)

        # Simulate upstream gradient with different values
        output_grad = np.array([[2.0, 5.0, 3.0]])
        grad = relu.backward(output_grad)

        # Only positive inputs should pass gradient
        expected = np.array([[2.0, 0.0, 3.0]])
        np.testing.assert_array_equal(grad, expected)

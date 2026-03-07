import numpy as np

from src.layers.activation.sigmoid import Sigmoid


class TestSigmoid:
    """Test suite for Sigmoid activation layer."""

    def test_forward_zero(self):
        """Test forward pass at zero (should be 0.5)."""
        sigmoid = Sigmoid()
        x = np.array([[0.0]])
        output = sigmoid(x)
        np.testing.assert_almost_equal(output, 0.5)

    def test_forward_positive_values(self):
        """Test forward pass with positive values."""
        sigmoid = Sigmoid()
        x = np.array([[1.0, 2.0, 10.0]])
        output = sigmoid(x)

        # All outputs should be between 0.5 and 1
        assert np.all(output >= 0.5)
        assert np.all(output <= 1.0)

        # Larger values should give outputs closer to 1
        assert output[0, 2] > output[0, 0]  # sigmoid(10) > sigmoid(1)

    def test_forward_negative_values(self):
        """Test forward pass with negative values."""
        sigmoid = Sigmoid()
        x = np.array([[-1.0, -2.0, -10.0]])
        output = sigmoid(x)

        # All outputs should be between 0 and 0.5
        assert np.all(output >= 0.0)
        assert np.all(output <= 0.5)

        # More negative values should give outputs closer to 0
        assert output[0, 2] < output[0, 0]  # sigmoid(-10) < sigmoid(-1)

    def test_forward_symmetry(self):
        """Test sigmoid symmetry: sigmoid(x) + sigmoid(-x) = 1."""
        sigmoid = Sigmoid()
        x = np.array([[1.0, 2.0, 5.0]])
        output_pos = sigmoid(x)

        sigmoid_neg = Sigmoid()
        output_neg = sigmoid_neg(-x)

        # sigmoid(x) + sigmoid(-x) should equal 1
        np.testing.assert_almost_equal(output_pos + output_neg, 1.0)

    def test_forward_shape_preservation(self):
        """Test that forward pass preserves input shape."""
        sigmoid = Sigmoid()
        x = np.random.randn(10, 5)
        output = sigmoid(x)
        assert output.shape == x.shape

    def test_forward_output_range(self):
        """Test that output is always in (0, 1)."""
        sigmoid = Sigmoid()
        x = np.random.randn(100, 100) * 10  # Large range
        output = sigmoid(x)
        assert np.all(output > 0.0)
        assert np.all(output <= 1.0)

    def test_forward_stores_output(self):
        """Test that forward pass stores the output for backward pass."""
        sigmoid = Sigmoid()
        x = np.array([[1.0, 2.0]])
        output = sigmoid(x)
        np.testing.assert_array_equal(sigmoid.output, output)

    def test_backward_at_zero(self):
        """Test backward pass at zero (max gradient = 0.25)."""
        sigmoid = Sigmoid()
        x = np.array([[0.0]])
        sigmoid(x)
        output_grad = np.array([[1.0]])
        grad = sigmoid.backward(output_grad)
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        np.testing.assert_almost_equal(grad[0, 0], 0.25)

    def test_backward_positive_values(self):
        """Test backward pass with positive values."""
        sigmoid = Sigmoid()
        x = np.array([[1.0, 2.0]])
        sigmoid(x)
        output_grad = np.array([[1.0, 1.0]])
        grad = sigmoid.backward(output_grad)

        # Gradient should be positive but decreasing for larger x
        assert np.all(grad > 0)
        assert grad[0, 0] > grad[0, 1]  # sigmoid'(1) > sigmoid'(2)

    def test_backward_negative_values(self):
        """Test backward pass with negative values."""
        sigmoid = Sigmoid()
        x = np.array([[-1.0, -2.0]])
        sigmoid(x)
        output_grad = np.array([[1.0, 1.0]])
        grad = sigmoid.backward(output_grad)

        # Gradient should be positive
        assert np.all(grad > 0)
        # More negative values should have smaller gradients
        # sigmoid'(-1) > sigmoid'(-2)
        assert grad[0, 0] > grad[0, 1]

    def test_backward_gradient_vanishing(self):
        """Test that gradient vanishes for extreme values."""
        sigmoid = Sigmoid()
        x = np.array([[100.0, -100.0]])
        sigmoid(x)
        output_grad = np.array([[1.0, 1.0]])
        grad = sigmoid.backward(output_grad)

        # Gradients should be very close to zero
        assert np.all(grad < 1e-10)

    def test_backward_shape_preservation(self):
        """Test that backward pass preserves gradient shape."""
        sigmoid = Sigmoid()
        x = np.random.randn(10, 5)
        sigmoid(x)
        output_grad = np.ones_like(x)
        grad = sigmoid.backward(output_grad)
        assert grad.shape == output_grad.shape

    def test_backward_batch(self):
        """Test backward pass with batch input."""
        sigmoid = Sigmoid()
        x = np.array([[1.0, 2.0], [-1.0, -2.0]])
        sigmoid(x)
        output_grad = np.ones_like(x)
        grad = sigmoid.backward(output_grad)

        # Check symmetry: sigmoid'(x) = sigmoid'(-x)
        np.testing.assert_almost_equal(grad[0, 0], grad[1, 0])
        np.testing.assert_almost_equal(grad[0, 1], grad[1, 1])

    def test_backward_scaled_gradient(self):
        """Test backward pass with scaled upstream gradient."""
        sigmoid = Sigmoid()
        x = np.array([[0.0]])
        sigmoid(x)

        output_grad_1 = np.array([[1.0]])
        output_grad_2 = np.array([[2.0]])

        grad_1 = sigmoid.backward(output_grad_1)
        grad_2 = sigmoid.backward(output_grad_2)

        # Gradient should scale linearly with upstream gradient
        np.testing.assert_almost_equal(grad_2 / grad_1, 2.0)

    def test_mathematical_correctness(self):
        """Test mathematical correctness of sigmoid derivative."""
        sigmoid = Sigmoid()
        x = np.array([[0.5, 1.0, 1.5]])
        output = sigmoid(x)

        # Manually compute expected gradient
        expected_sigmoid = 1 / (1 + np.exp(-x))
        expected_grad = expected_sigmoid * (1 - expected_sigmoid)

        sigmoid_layer2 = Sigmoid()
        sigmoid_layer2(x)
        output_grad = np.ones_like(x)
        grad = sigmoid_layer2.backward(output_grad)

        np.testing.assert_almost_equal(grad, expected_grad)

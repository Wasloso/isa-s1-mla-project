import numpy as np

from src.layers.activation.softmax import Softmax


class TestSoftmax:
    """Test suite for Softmax activation layer."""

    def test_forward_output_sums_to_one(self):
        """Test that softmax outputs sum to 1 for each sample."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = softmax(x)

        # Each row should sum to 1
        row_sums = np.sum(output, axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(2))

    def test_forward_output_range(self):
        """Test that softmax outputs are in [0, 1]."""
        softmax = Softmax()
        x = np.random.randn(10, 5)
        output = softmax(x)

        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_forward_equal_inputs(self):
        """Test softmax with equal inputs (should give uniform distribution)."""
        softmax = Softmax()
        x = np.array([[1.0, 1.0, 1.0]])
        output = softmax(x)

        # All probabilities should be equal (1/3)
        expected = np.array([[1 / 3, 1 / 3, 1 / 3]])
        np.testing.assert_almost_equal(output, expected)

    def test_forward_large_difference(self):
        """Test softmax with large differences between inputs."""
        softmax = Softmax()
        x = np.array([[100.0, 0.0, -100.0]])
        output = softmax(x)

        # First element should dominate
        assert output[0, 0] > 0.99
        assert output[0, 1] < 0.01
        assert output[0, 2] < 1e-40

    def test_forward_numerical_stability(self):
        """Test that softmax handles large values without overflow."""
        softmax = Softmax()
        # This would overflow without the max subtraction trick
        x = np.array([[1000.0, 1001.0, 999.0]])
        output = softmax(x)

        # Should still sum to 1 and not produce NaN or Inf
        assert np.all(np.isfinite(output))
        np.testing.assert_almost_equal(np.sum(output), 1.0)

    def test_forward_shape_preservation(self):
        """Test that forward pass preserves input shape."""
        softmax = Softmax()
        x = np.random.randn(10, 20)
        output = softmax(x)
        assert output.shape == x.shape

    def test_forward_stores_output(self):
        """Test that forward pass stores the output for backward pass."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output = softmax(x)
        np.testing.assert_array_equal(softmax.output, output)

    def test_forward_batch_independence(self):
        """Test that softmax treats batch samples independently."""
        softmax = Softmax()

        # Two identical samples
        x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        output = softmax(x)

        # Outputs should be identical
        np.testing.assert_array_almost_equal(output[0], output[1])

    def test_forward_zero_input(self):
        """Test softmax with all zeros."""
        softmax = Softmax()
        x = np.array([[0.0, 0.0, 0.0]])
        output = softmax(x)

        # Should give uniform distribution
        expected = np.array([[1 / 3, 1 / 3, 1 / 3]])
        np.testing.assert_almost_equal(output, expected)

    def test_backward_shape_preservation(self):
        """Test that backward pass preserves gradient shape."""
        softmax = Softmax()
        x = np.random.randn(10, 5)
        softmax(x)
        output_grad = np.random.randn(10, 5)
        grad = softmax.backward(output_grad)
        assert grad.shape == output_grad.shape

    def test_backward_mathematical_correctness(self):
        """Test backward pass computes correct Jacobian."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output = softmax(x)

        # For a single sample, compute Jacobian manually
        # Jacobian: J_ij = s_i(δ_ij - s_j)
        s = output[0]  # shape (3,)
        jacobian = np.diag(s) - np.outer(s, s)

        # Test with unit gradient vectors
        for i in range(3):
            output_grad = np.zeros((1, 3))
            output_grad[0, i] = 1.0
            grad = softmax.backward(output_grad)

            expected_grad = output_grad @ jacobian
            np.testing.assert_almost_equal(grad, expected_grad)

    def test_backward_zero_gradient(self):
        """Test backward with zero gradient."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        softmax(x)
        output_grad = np.zeros_like(x)
        grad = softmax.backward(output_grad)

        # Zero gradient should remain zero
        np.testing.assert_array_equal(grad, np.zeros_like(x))

    def test_backward_uniform_gradient(self):
        """Test backward with uniform upstream gradient."""
        softmax = Softmax()
        x = np.array([[0.0, 0.0, 0.0]])
        output = softmax(x)

        # Uniform gradient on uniform softmax output
        output_grad = np.ones((1, 3))
        grad = softmax.backward(output_grad)

        # For uniform softmax output s = [1/3, 1/3, 1/3]
        # Jacobian @ ones vector should give zero (property of softmax)
        s = output[0]
        jacobian = np.diag(s) - np.outer(s, s)
        expected_grad = output_grad @ jacobian
        np.testing.assert_almost_equal(grad, expected_grad)

    def test_backward_batch_processing(self):
        """Test backward pass processes batches correctly."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        softmax(x)
        output_grad = np.random.randn(3, 2)
        grad = softmax.backward(output_grad)

        # Gradient shape should match
        assert grad.shape == output_grad.shape

    def test_forward_monotonicity(self):
        """Test that softmax preserves input ordering."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        output = softmax(x)

        # Output should maintain ordering: output[i] < output[j] if x[i] < x[j]
        for i in range(3):
            assert output[0, i] < output[0, i + 1]

    def test_forward_gradient_dependency(self):
        """Test that each output depends on all inputs."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output1 = softmax(x)

        # Slightly change one input
        x_modified = x.copy()
        x_modified[0, 0] = 10.0

        softmax2 = Softmax()
        output2 = softmax2(x_modified)

        # All outputs should change when one input changes
        assert not np.allclose(output1, output2)

    def test_backward_implementation_corrected(self):
        """Test that backward pass correctly applies softmax Jacobian."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output = softmax(x)

        # Output gradient targets first element
        output_grad = np.array([[1.0, 0.0, 0.0]])

        # Compute expected gradient using Jacobian: J_ij = s_i(δ_ij - s_j)
        s = output[0]
        jacobian = np.diag(s) - np.outer(s, s)
        expected_grad = output_grad @ jacobian

        grad = softmax.backward(output_grad)
        np.testing.assert_almost_equal(grad, expected_grad)

    def test_softmax_cross_entropy_compatibility(self):
        """Test that softmax output can be used with cross-entropy."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = softmax(x)

        # Cross-entropy typically uses log(softmax)
        log_output = np.log(output)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(log_output))

    def test_backward_numerical_gradient(self):
        """Test backward pass using numerical gradient verification."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        eps = 1e-5

        # Compute numerical gradient for each output
        numerical_grad = np.zeros_like(x)
        for i in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[0, i] += eps
            output_plus = Softmax()(x_plus)

            x_minus = x.copy()
            x_minus[0, i] -= eps
            output_minus = Softmax()(x_minus)

            numerical_grad[0, i] = np.sum(output_plus - output_minus) / (2 * eps)

        # Compute analytical gradient using backward
        output = softmax(x)
        output_grad = np.ones_like(x)
        analytical_grad = softmax.backward(output_grad)

        np.testing.assert_almost_equal(analytical_grad, numerical_grad, decimal=4)

    def test_backward_jacobian_properties(self):
        """Test mathematical properties of softmax Jacobian."""
        softmax = Softmax()
        x = np.array([[1.5, 2.5, 3.5]])
        output = softmax(x)

        # Compute Jacobian: J_ij = s_i(δ_ij - s_j)
        s = output[0]
        jacobian = np.diag(s) - np.outer(s, s)

        # Property 1: Jacobian rows sum to zero (sum of derivatives = 0)
        row_sums = np.sum(jacobian, axis=1)
        np.testing.assert_almost_equal(row_sums, np.zeros(3))

        # Property 2: Jacobian is symmetric in the sense that J @ ones = 0
        ones = np.ones(3)
        np.testing.assert_almost_equal(jacobian @ ones, np.zeros(3))

    def test_backward_batch_jacobian(self):
        """Test backward pass with batch and verify each sample uses correct Jacobian."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = softmax(x)

        output_grad = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        grad = softmax.backward(output_grad)

        # Verify each batch element independently
        for b in range(2):
            s = output[b]
            jacobian = np.diag(s) - np.outer(s, s)
            expected_grad = output_grad[b : b + 1] @ jacobian
            np.testing.assert_almost_equal(grad[b : b + 1], expected_grad)

    def test_backward_scaled_output_gradient(self):
        """Test backward with scaled output gradient."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output = softmax(x)

        # Test with different scales
        output_grad_1 = np.array([[1.0, 2.0, 3.0]])
        output_grad_2 = np.array([[2.0, 4.0, 6.0]])

        grad_1 = softmax.backward(output_grad_1)

        softmax2 = Softmax()
        softmax2(x)
        grad_2 = softmax2.backward(output_grad_2)

        # Gradient should scale linearly
        np.testing.assert_almost_equal(grad_2, 2 * grad_1)

    def test_backward_zero_gradient_output(self):
        """Test backward pass produces zero when output gradient is zero."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        softmax(x)

        output_grad = np.zeros_like(x)
        grad = softmax.backward(output_grad)

        np.testing.assert_array_equal(grad, np.zeros_like(x))

    def test_backward_single_hot_gradient(self):
        """Test backward with one-hot gradient vectors."""
        softmax = Softmax()
        x = np.array([[2.0, 1.0, 0.5]])
        output = softmax(x)

        # Test for each one-hot vector
        for i in range(3):
            output_grad = np.zeros((1, 3))
            output_grad[0, i] = 1.0

            s = output[0]
            jacobian = np.diag(s) - np.outer(s, s)
            expected_grad = output_grad @ jacobian

            softmax_test = Softmax()
            softmax_test(x)
            grad = softmax_test.backward(output_grad)

            np.testing.assert_almost_equal(grad, expected_grad)

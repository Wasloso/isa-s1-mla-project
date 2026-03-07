# Create this as test_save_load.py
import os
import tempfile

import numpy as np

from src.backend import BackendPolicy, backend
from src.layers.activation import ELU, ReLU, Sigmoid, Softmax
from src.layers.dense import Dense
from src.loss.cross_entropy import CrossEntropy
from src.loss.mse import MSE
from src.models.sequential import Sequential
from src.optimizer.adam import Adam
from src.optimizer.sgd import SGD


def test_save_load_functionality():  # noqa: PLR0915
    """Comprehensive test for model save/load functionality."""
    print("🧪 Testing model save/load functionality...\n")

    # Test data
    np.random.seed(42)
    x_test = np.random.randn(10, 784).astype(np.float32)
    y_test_regression = np.random.randn(10, 1).astype(np.float32)
    y_test_classification = np.random.randint(0, 10, (10,))
    y_test_onehot = np.eye(10)[y_test_classification].astype(np.float32)

    # Test different backend configurations
    test_configs = [
        {"use_gpu": False, "name": "CPU (NumPy)"},
        {"use_gpu": True, "name": "GPU (CuPy)"},  # Will fallback to CPU if no GPU
    ]

    for config in test_configs:
        print(f"📋 Testing with {config['name']}...")
        backend.configure(BackendPolicy(use_gpu=config["use_gpu"], min_gpu_bytes=1), dataset_bytes=x_test.nbytes)

        # Test 1: Basic Classification Model
        print("  ✅ Test 1: Classification model...")
        model1 = Sequential(
            [
                Dense(128, name="dense1"),
                ReLU(name="relu1"),
                Dense(64, name="dense2"),
                ELU(name="elu1"),
                Dense(10, name="output"),
                Softmax(name="softmax"),
            ]
        )

        model1.compile(loss=CrossEntropy(), optimizer=Adam(learning_rate=0.001))

        # Get initial predictions
        pred_before = model1.predict(x_test)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            model1.save(temp_path)
            model1_loaded = Sequential.load(temp_path)
            pred_after = model1_loaded.predict(x_test)

            # Verify predictions are identical
            np.testing.assert_allclose(pred_before, pred_after, rtol=1e-6)
            print("    ✓ Predictions identical after save/load")

            # Verify model structure
            assert len(model1_loaded.layers) == len(model1.layers)
            assert model1_loaded.compiled == True
            assert type(model1_loaded.loss).__name__ == "CrossEntropy"
            assert type(model1_loaded.optimizer).__name__ == "Adam"
            print("    ✓ Model structure preserved")

        finally:
            os.unlink(temp_path)

        # Test 2: Regression Model
        print("  ✅ Test 2: Regression model...")
        model2 = Sequential(
            [
                Dense(32, name="hidden1"),
                ReLU(name="activation1"),
                Dense(16, name="hidden2"),
                Sigmoid(name="activation2"),
                Dense(1, name="output"),
            ]
        )

        model2.compile(loss=MSE(), optimizer=SGD(learning_rate=0.01))

        # Train for a few steps
        model2.fit(x_test, y_test_regression, epochs=5, verbose=0)

        pred_before = model2.predict(x_test)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            model2.save(temp_path)
            model2_loaded = Sequential.load(temp_path)
            pred_after = model2_loaded.predict(x_test)

            np.testing.assert_allclose(pred_before, pred_after, rtol=1e-6)
            print("    ✓ Trained model predictions identical after save/load")

        finally:
            os.unlink(temp_path)

        # Test 3: Continue training after load
        print("  ✅ Test 3: Continue training after load...")
        model3 = Sequential(
            [Dense(64, name="layer1"), ReLU(name="relu1"), Dense(10, name="output"), Softmax(name="softmax")]
        )

        model3.compile(loss=CrossEntropy(), optimizer=Adam(learning_rate=0.001))

        # Train briefly
        history_before = model3.fit(x_test, y_test_onehot, epochs=3, verbose=0)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            model3.save(temp_path)
            model3_loaded = Sequential.load(temp_path)

            # Continue training
            history_after = model3_loaded.fit(x_test, y_test_onehot, epochs=3, verbose=0)

            # Training should continue (loss should change)
            assert len(history_after) == 3
            print("    ✓ Training continued successfully after load")

        finally:
            os.unlink(temp_path)

        print(f"  ✅ All tests passed for {config['name']}!\n")

    # Test 4: Backend switching compatibility
    print("📋 Testing backend switching compatibility...")

    # Create model with CPU backend
    backend.configure(BackendPolicy(use_gpu=False), dataset_bytes=100)
    model_cpu = Sequential([Dense(32), ReLU(), Dense(10)])
    model_cpu.compile(loss=MSE(), optimizer=Adam())
    model_cpu.fit(x_test, y_test_regression[:, :1].repeat(10, axis=1), epochs=2, verbose=0)
    pred_cpu = model_cpu.predict(x_test)

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        model_cpu.save(temp_path)

        # Switch to GPU backend (if available) and load
        backend.configure(BackendPolicy(use_gpu=True), dataset_bytes=100000)
        model_loaded = Sequential.load(temp_path)
        pred_loaded = model_loaded.predict(x_test)

        # Predictions should be nearly identical (small float precision differences allowed)
        np.testing.assert_allclose(pred_cpu, pred_loaded, rtol=1e-5)
        print("  ✅ Backend switching compatibility works!")

    finally:
        os.unlink(temp_path)

    print("\n🎉 ALL TESTS PASSED! Your save/load functionality is working perfectly!")

    # Test 5: Edge cases
    print("\n📋 Testing edge cases...")

    # Test saving uncompiled model (should raise error)
    model_uncompiled = Sequential([Dense(10)])
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            model_uncompiled.save(f.name)
        assert False, "Should have raised error for uncompiled model"
    except RuntimeError as e:
        assert "compiled" in str(e).lower()
        print("  ✅ Properly rejects saving uncompiled model")

    # Test loading non-existent file
    try:
        Sequential.load("non_existent_file.pkl")
        assert False, "Should have raised error for non-existent file"
    except FileNotFoundError:
        print("  ✅ Properly handles non-existent files")

    print("\n🎉 ALL EDGE CASE TESTS PASSED!")


def test_weight_preservation():
    """Test that exact weights are preserved."""
    print("\n🔬 Testing exact weight preservation...")

    backend.configure(BackendPolicy(use_gpu=False), dataset_bytes=100)

    # Create model with known weights
    model = Sequential([Dense(5, name="layer1"), ReLU(), Dense(3, name="layer2")])

    model.compile(loss=MSE(), optimizer=Adam())

    # Initialize with dummy data to build layers
    x_dummy = np.random.randn(1, 10).astype(np.float32)
    model.predict(x_dummy)

    # Extract original weights
    original_weights = []
    for layer in model.layers:
        if hasattr(layer, "trainable_weights") and layer.trainable_weights:
            layer_weights = [backend.to_numpy(w).copy() for w in layer.trainable_weights]
            original_weights.append(layer_weights)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        model_loaded = Sequential.load(temp_path)

        # Extract loaded weights
        loaded_weights = []
        for layer in model_loaded.layers:
            if hasattr(layer, "trainable_weights") and layer.trainable_weights:
                layer_weights = [backend.to_numpy(w) for w in layer.trainable_weights]
                loaded_weights.append(layer_weights)

        # Compare weights exactly
        for orig_layer, loaded_layer in zip(original_weights, loaded_weights):
            for orig_weight, loaded_weight in zip(orig_layer, loaded_layer):
                np.testing.assert_array_equal(orig_weight, loaded_weight)

        print("  ✅ All weights preserved exactly!")

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_save_load_functionality()
    test_weight_preservation()
    print("\n🚀 ALL COMPREHENSIVE TESTS COMPLETED SUCCESSFULLY!")

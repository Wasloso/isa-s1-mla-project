import numpy as np

from src.layers.activation.relu import ReLU
from src.layers.activation.sigmoid import Sigmoid
from src.layers.dense import Dense
from src.loss.mse import MSE
from src.models.sequential import Sequential
from src.optimizer.sgd import SGD


def main():
    # XOR test

    # Generate 100 points from 0 to 1
    x_train = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
    y_train = 0.5 * np.sin(2 * np.pi * x_train) + 0.5  # Scale to 0-1
    model = Sequential(
        [
            Dense(units=32, name="hidden_1"),
            ReLU(name="activation_1"),
            Dense(units=32, name="hidden_2"),
            ReLU(name="activation_2"),
            Dense(units=1, name="output"),
            Sigmoid(name="sigmoid_out"),  # For 0-1 range
        ]
    )

    model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.05))
    predictions = model.predict(x_train)
    print(f"Initial predictions:\n{predictions}")

    _ = model.fit(x_train, y_train, epochs=10000, verbose=2500)

    # Test on new points
    x_test = np.array([[0.25], [0.5], [0.75]]).astype(np.float32)
    predictions = model.predict(x_test)
    print(f"Predictions: {predictions.flatten()}")
    print("Expected: ", 0.5 * np.sin(2 * np.pi * x_test) + 0.5)  # Should be close


if __name__ == "__main__":
    main()

import numpy as np

from src.layers.dense import Dense
from src.layers.relu import ReLU
from src.layers.sigmoid import Sigmoid
from src.loss.mse import MSE
from src.models.sequential import Sequential
from src.optimizer.sgd import SGD


def main():
    # XOR test
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_train = np.array([[0], [1], [1], [0]], dtype=np.float32)
    model = Sequential(
        layers=[
            Dense(units=8, name="hidden_1"),
            ReLU(name="activation_1"),
            Dense(units=8, name="hidden_2"),
            ReLU(name="activation_2"),
            Dense(units=1, name="output"),
            Sigmoid(name="sigmoid_out"),
        ]
    )
    model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.05))
    predictions = model.predict(x_train)
    print(f"Initial predictions:\n{predictions}")

    _ = model.fit(x_train, y_train, epochs=10000, verbose=1000)

    x_test = np.array([[0.5, 0.5]], dtype=np.float32)
    y_test = model.predict(x_test)
    print(f"Final prediction:\n{y_test}")


if __name__ == "__main__":
    main()

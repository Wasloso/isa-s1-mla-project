import pickle
import time
from abc import ABC, abstractmethod

from src.backend import backend
from src.loss.base import Loss
from src.optimizer.base import Optimizer
from src.types import Array


class Network(ABC):
    def __init__(
        self,
    ) -> None:
        self.compiled = False

    def predict(self, x: Array, training: bool = False, mask: Array | None = None):
        x_dev = backend.asarray(x)
        mask_dev = backend.asarray(mask) if mask is not None else None
        pred_dev = self._predict(x_dev, training=training, mask=mask_dev)
        return backend.to_numpy(pred_dev)

    @abstractmethod
    def _predict(self, x: Array, training: bool = False, mask: Array | None = None):
        pass

    @abstractmethod
    def train_step(self, x: Array, y: Array, mask: Array | None = None) -> float:
        pass

    def compile(self, loss: Loss, optimizer: Optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.compiled = True

    @abstractmethod
    def reset(self):
        if self.optimizer is not None:
            self.optimizer.reset()

    def fit(
        self,
        x_train: Array,
        y_train: Array,
        epochs: int,
        verbose: int = 10,
        batch_size: int | None = None,
        mask: Array | None = None,
        lr_decay: float | None = None,
        lr_decay_epochs: int | None = None,
    ):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training.")
        self.reset()
        history = []
        x_dev = backend.asarray(x_train)
        y_dev = backend.asarray(y_train)
        num_samples = x_dev.shape[0]
        mask_dev = backend.asarray(mask) if mask is not None else None

        start_training_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            if lr_decay is not None and lr_decay_epochs is not None and epoch > 0 and epoch % lr_decay_epochs == 0:
                self.optimizer.learning_rate *= lr_decay
                if verbose > 0:
                    print(f"Epoch {epoch}: Learning rate decayed to {self.optimizer.learning_rate:.6f}")
            if batch_size is not None:
                epoch_loss = 0.0
                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    x_batch = x_dev[start:end]
                    y_batch = y_dev[start:end]
                    m_batch = mask_dev[start:end] if mask_dev is not None else None
                    epoch_loss += self.train_step(x_batch, y_batch, m_batch)
                loss_val = epoch_loss / (num_samples / batch_size)
            else:
                loss_val = self.train_step(x_dev, y_dev, mask_dev)
            history.append(float(loss_val))
            epoch_duration = time.time() - epoch_start_time
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                epochs_done = epoch + 1
                epochs_remaining = epochs - epochs_done
                total_elapsed = time.time() - start_training_time
                avg_time_per_epoch = total_elapsed / epochs_done
                eta_seconds = epochs_remaining * avg_time_per_epoch
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                print(
                    f"Epoch {epochs_done}/{epochs} - Loss: {loss_val:.6f} "
                    f"- {epoch_duration:.2f}s/epoch - ETA: {eta_str}"
                )

        total_time = time.time() - start_training_time
        if verbose > 0:
            print(f"\nTraining Complete. Total Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        return history, total_time

    def save(self, filepath: str) -> None:
        if not self.compiled:
            raise RuntimeError("Model must be compiled before saving.")
        self._convert_to_numpy()

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        self._convert_to_current_backend()

    def _convert_to_numpy(self):
        for layer in getattr(self, "layers", []):
            if hasattr(layer, "trainable_weights"):
                layer.trainable_weights = [backend.to_numpy(w) for w in layer.trainable_weights]

    def _convert_to_current_backend(self):
        for layer in getattr(self, "layers", []):
            if hasattr(layer, "trainable_weights"):
                layer.trainable_weights = [backend.asarray(w) for w in layer.trainable_weights]

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        model._convert_to_current_backend()
        return model

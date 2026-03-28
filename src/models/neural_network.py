import os
import pickle
import time
from abc import ABC, abstractmethod

from src.backend import backend
from src.loss.base import Loss
from src.models.history import TrainingHistory
from src.optimizer.base import Optimizer
from src.types import Array


class Network(ABC):
    def __init__(
        self,
    ) -> None:
        self.compiled = False
        self._activity_labels = []

    def predict(self, x: Array, training: bool = False, mask: Array | None = None):
        x_dev = backend.asarray(x, dtype=self.xp.float32)
        mask_dev = backend.asarray(mask, dtype=self.xp.float32) if mask is not None else None
        pred_dev = self._predict(x_dev, training=training, mask=mask_dev)
        return backend.to_numpy(pred_dev)

    @abstractmethod
    def _predict(self, x: Array, training: bool = False, mask: Array | None = None):
        pass

    @abstractmethod
    def train_step(self, x: Array, y: Array, mask: Array | None = None) -> tuple[float, Array]:
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
        x_val: Array | None = None,
        y_val: Array | None = None,
        verbose: int = 10,
        batch_size: int | None = None,
        mask: Array | None = None,
        val_mask: Array | None = None,
        lr_decay: float | None = None,
        lr_decay_epochs: int | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        reset: bool = False,
    ):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training.")
        if reset:
            self.reset()
        history = TrainingHistory()
        x_dev = backend.asarray(x_train, dtype=self.xp.float32)
        y_dev = backend.asarray(y_train)
        if x_val is not None and y_val is not None:
            x_val_dev = backend.asarray(x_val)
            y_val_dev = backend.asarray(y_val)
            val_mask_dev = backend.asarray(val_mask) if val_mask is not None else None
        num_samples = x_dev.shape[0]
        mask_dev = backend.asarray(mask) if mask is not None else None
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        start_training_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            if lr_decay is not None and lr_decay_epochs is not None and epoch > 0 and epoch % lr_decay_epochs == 0:
                self.optimizer.learning_rate *= lr_decay
                if verbose > 0:
                    print(f"Epoch {epoch}: Learning rate decayed to {self.optimizer.learning_rate:.6f}")
            if batch_size is not None:
                epoch_loss = 0.0
                epoch_correct = 0
                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    x_batch = x_dev[start:end]
                    y_batch = y_dev[start:end]
                    m_batch = mask_dev[start:end] if mask_dev is not None else None
                    loss_batch, pred_batch = self.train_step(x_batch, y_batch, m_batch)
                    epoch_loss += loss_batch
                    if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
                        correct = self.xp.sum(self.xp.argmax(pred_batch, axis=1) == self.xp.argmax(y_batch, axis=1))
                    else:
                        correct = self.xp.sum(self.xp.argmax(pred_batch, axis=1) == y_batch.reshape(-1))
                    epoch_correct += int(correct)
                loss_val = epoch_loss / (num_samples / batch_size)
                train_acc = epoch_correct / num_samples
            else:
                loss_val, pred = self.train_step(x_dev, y_dev, mask_dev)
                train_acc = float(self.xp.mean(self.xp.argmax(pred, axis=1) == y_dev))

            val_loss, val_acc = None, None
            if x_val is not None and y_val is not None:
                val_pred = self._predict(x_val_dev, training=False, mask=val_mask_dev)
                val_loss = float(self.loss.forward(val_pred, y_val_dev))
                if len(y_val_dev.shape) > 1 and y_val_dev.shape[1] > 1:
                    val_acc = float(self.xp.mean(self.xp.argmax(val_pred, axis=1) == self.xp.argmax(y_val_dev, axis=1)))
                else:
                    val_acc = float(self.xp.mean(self.xp.argmax(val_pred, axis=1) == y_val_dev.reshape(-1)))
            epoch_duration = time.time() - epoch_start_time
            history.update(
                train_loss=loss_val,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=self.optimizer.learning_rate,
                epoch_time=epoch_duration,
            )
            if early_stopping_patience is not None and val_loss is not None:
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    self.save("temp_early_stopping_best.nnp")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose > 0:
                            print(f"\nEarly stopping triggered at epoch {epoch}! Restoring best weights.")
                        best_model = self.load("temp_early_stopping_best.nnp")
                        self.layers = best_model.layers
                        if os.path.exists("temp_early_stopping_best.nnp"):
                            os.remove("temp_early_stopping_best.nnp")

                        break
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                epochs_done = epoch + 1
                epochs_remaining = epochs - epochs_done
                total_elapsed = time.time() - start_training_time
                avg_time_per_epoch = total_elapsed / epochs_done
                eta_seconds = epochs_remaining * avg_time_per_epoch
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                print(
                    f"Epoch {epochs_done}/{epochs}, {epoch_duration:.2f}s/epoch - ETA: {eta_str}\n"
                    f"Loss: {loss_val:.6f}, Acc: {train_acc:.4f}"
                    f"{f'\nVal Loss: {val_loss:.6f} - Val Acc: {val_acc:.4f}' if val_loss is not None and val_acc is not None else ''}"
                )

        if early_stopping_patience is not None:
            history.best_epoch = best_epoch

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
        converted = {}
        cache_vars = ["x_padded", "windows", "input", "x_centered", "std_inv", "x_norm"]

        for layer in getattr(self, "layers", []):
            for k, v in layer.__dict__.items():
                if k in cache_vars:
                    setattr(layer, k, None)
                    continue
                if type(v).__name__ == "ndarray" or type(v).__module__.startswith("cupy"):
                    vid = id(v)
                    if vid not in converted:
                        converted[vid] = backend.to_numpy(v)
                    setattr(layer, k, converted[vid])
            if hasattr(layer, "trainable_weights"):
                new_weights = []
                for w in layer.trainable_weights:
                    wid = id(w)
                    if wid not in converted:
                        converted[wid] = backend.to_numpy(w)
                    new_weights.append(converted[wid])
                layer.trainable_weights = new_weights

    def _convert_to_current_backend(self):
        converted = {}
        for layer in getattr(self, "layers", []):
            for k, v in layer.__dict__.items():
                if type(v).__name__ == "ndarray" or type(v).__module__.startswith("cupy"):
                    vid = id(v)
                    if vid not in converted:
                        converted[vid] = backend.asarray(backend.to_numpy(v))
                    setattr(layer, k, converted[vid])
            if hasattr(layer, "trainable_weights"):
                new_weights = []
                for w in layer.trainable_weights:
                    wid = id(w)
                    if wid not in converted:
                        converted[wid] = backend.asarray(backend.to_numpy(w))
                    new_weights.append(converted[wid])
                layer.trainable_weights = new_weights

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        model._convert_to_current_backend()
        return model

    @property
    def scaler(self):
        return getattr(self, "_scaler", None)

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def activity_labels(self):
        return self._activity_labels

    @activity_labels.setter
    def activity_labels(self, value: list[str]):
        self._activity_labels = value

    @property
    def xp(self):
        return backend.xp

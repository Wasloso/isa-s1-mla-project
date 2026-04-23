class TrainingHistory:
    def __init__(self):
        self.metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "epoch_time": []}
        self.epoch_times = []
        self.best_epoch: int | None = None

    def update(
        self,
        train_loss: float,
        train_acc: float,
        lr: float,
        epoch_time: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ):
        self.metrics["train_loss"].append(float(train_loss))
        self.metrics["train_acc"].append(float(train_acc))
        self.metrics["lr"].append(float(lr))
        self.metrics["epoch_time"].append(float(epoch_time))

        if val_loss is not None:
            self.metrics["val_loss"].append(float(val_loss))
        if val_acc is not None:
            self.metrics["val_acc"].append(float(val_acc))

    def __getitem__(self, key):
        return self.metrics[key]

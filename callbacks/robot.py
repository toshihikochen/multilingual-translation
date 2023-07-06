from lightning.pytorch.callbacks import Callback
import requests
import warnings


class Robot(Callback):
    def __init__(self, url, interval=None):
        super().__init__()
        self.url = url

        self.interval = interval
        self.last_sent = 0

    def send_message(self, text):
        headers = {'Content-Type': 'application/json'}
        data = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }
        requests.post(self.url, headers=headers, json=data)
        
    def on_train_start(self, trainer, pl_module):
        if isinstance(self.interval, int):
            if self.interval > trainer.num_training_batches:
                warnings.warn(f"interval {self.interval} is larger than the number of training batches {trainer.num_training_batches}.")
                self.interval = None
            if self.interval < 1:
                warnings.warn(f"interval {self.interval} is smaller than 1. Ignored.")
                self.interval = None
        elif isinstance(self.interval, float):
            if self.interval > 1.0 or self.interval < 0.0:
                warnings.warn(f"interval {self.interval} is larger than 1.0 or less than 0.0. Ignored.")
                self.interval = None
            else:
                self.interval = int(self.interval * trainer.num_training_batches)
        else:
            warnings.warn(f"interval {self.interval} is not a valid number. Ignored.")
            self.interval = None

        if trainer.fast_dev_run:
            self.send_message("Train *test* started")
        else:
            self.send_message("Train started")

    def on_train_end(self, trainer, pl_module):
        if trainer.fast_dev_run:
            self.send_message("Train *test* ended")
        else:
            self.send_message("Train ended")

    def on_train_epoch_end(self, trainer, pl_module):
        self.send_message(f"Epoch {trainer.current_epoch} ended")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.interval is None:
            return
        if batch_idx % self.interval < self.interval - 1:
            return

        self.send_message(f"Reach Epoch {trainer.current_epoch}/{trainer.max_epochs} "
                          f"Batch {batch_idx}/{trainer.num_training_batches}")

    def on_validation_start(self, trainer, pl_module):
        self.send_message("Validation started")

    def on_validation_end(self, trainer, pl_module):
        outputs = trainer.callback_metrics
        text = f"Validation ended\n"
        for key, value in outputs.items():
            text += f"{key}: {value}\n"
        self.send_message(text)

    def on_exception(self, trainer, pl_module, exception):
        self.send_message(f"Caught exception: \n{exception}")

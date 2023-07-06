import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path, write_interval):
        super().__init__(write_interval)
        # create a file to store the predictions
        self.output_path = output_path
        with open(self.output_path, 'w') as f:
            f.write('')

    def write_on_batch_end(self, trainer,  pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx=0):
        # write the predictions to the file
        with open(self.output_path, 'a') as f:
            f.write(prediction[0] + '\n')

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # write the predictions to the file
        with open(self.output_path, 'a') as f:
            for prediction in predictions:
                f.write(prediction[0] + '\n')

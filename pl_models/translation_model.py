import torch
import lightning as pl
import transformers

from metrics import BLEU, TER

optimizers = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adafactor': transformers.Adafactor,
}

schedulers = {
    'linear': torch.optim.lr_scheduler.LinearLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


class TranslationModel(pl.LightningModule):
    def __init__(
            self, tokenizer, model, optimizer=None, lr=None, scheduler=None, optim_kwargs=None, scheduler_kwargs=None,
            scheduler_interval='epoch'
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_interval = scheduler_interval

        self.save_hyperparameters(ignore=['tokenizer', 'model'])

        self.bleu = BLEU(tokenize_fn=tokenizer.tokenize)
        self.ter = TER(tokenize_fn=tokenizer.tokenize)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        # calculate BLEU score
        labels, predictions = [], []
        for i in range(len(batch['labels'])):
            label, prediction = batch['labels'][i], outputs.logits[i].argmax(dim=-1)
            labels.append(label[label != -100])
            predictions.append(prediction[prediction != -100])
        self.log_dict(
            {
                'val_loss': outputs.loss,
                'bleu': self.bleu(
                    self.tokenizer.batch_decode(predictions, skip_special_tokens=True),
                    [self.tokenizer.batch_decode(labels, skip_special_tokens=True)],
                ),
                'ter': self.ter(
                    self.tokenizer.batch_decode(predictions, skip_special_tokens=True),
                    [self.tokenizer.batch_decode(labels, skip_special_tokens=True)],
                ),
            }, prog_bar=True
        )
        return outputs.loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model.generate(**batch)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def configure_optimizers(self):
        if self.optim_kwargs is None:
            self.optim_kwargs = {}
        optimizer = optimizers[self.optimizer](self.parameters(), lr=self.lr, **self.optim_kwargs)

        if self.scheduler is not None:
            scheduler = schedulers[self.scheduler](optimizer, **self.scheduler_kwargs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': self.scheduler_interval,
                    'frequency': 1,
                }
            }

        return {'optimizer': optimizer}

import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
import lightning as pl
import transformers

from datasets import ParquetMatrix


class MatrixDataModule(pl.LightningDataModule):
    def __init__(self, train_data_paths, tokenizer, test_data_paths=None, batch_size=4, val_split=None,
                 specific_lang_pair=None, use_augment=False):
        super().__init__()
        self.train_data_paths = train_data_paths
        self.test_data_paths = test_data_paths
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.val_split = val_split
        self.specific_lang_pair = specific_lang_pair
        self.use_augment = use_augment

        self.lang_pairs = []    # [(src, tgt), ...]
        if self.specific_lang_pair:
            self.lang_pairs = [specific_lang_pair]
        else:
            parquet_datasets = [
                ParquetMatrix(
                    data_path, tokenizer=self.tokenizer, use_augment=self.use_augment,
                    specific_lang_pair=self.specific_lang_pair
                )
                for data_path in self.train_data_paths
            ]
            lang_set = set()
            for dataset in parquet_datasets:
                lang_set.update(dataset.get_langs())
            lang_set = list(lang_set)

            lang_pairs = []
            for i in range(len(lang_set) - 1):
                for j in range(i + 1, len(lang_set)):
                    lang_pairs.append((lang_set[i], lang_set[j]))
                    lang_pairs.append((lang_set[j], lang_set[i]))
            self.lang_pairs = lang_pairs

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            parquet_datasets = [
                ParquetMatrix(
                    data_path, tokenizer=self.tokenizer, use_augment=self.use_augment,
                    specific_lang_pair=self.specific_lang_pair
                )
                for data_path in self.train_data_paths
            ]
            dataset = ConcatDataset(parquet_datasets)
            if not self.val_split:
                # full train
                self.train_dataset = dataset
            else:
                if isinstance(self.val_split, float):
                    val_size = int(self.val_split * len(dataset))
                else:
                    val_size = self.val_split
                train_size = len(dataset) - val_size
                self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            if self.test_data_paths:
                parquet_datasets = [
                    ParquetMatrix(
                        data_path, tokenizer=self.tokenizer, use_augment=False,
                        specific_lang_pair=self.specific_lang_pair
                    )
                    for data_path in self.test_data_paths
                ]
                self.test_dataset = ConcatDataset(parquet_datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.get_collate_fn(),
            pin_memory=True
        )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=self.get_collate_fn(),
                pin_memory=True
            )
        return None

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=self.get_collate_fn(),
                pin_memory=True
            )
        return None

    def get_collate_fn(self):
        if self.tokenizer is None:
            return None
        else:
            return transformers.DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, pad_to_multiple_of=8, return_tensors='pt'
            )

    def get_lang_pairs(self):
        return self.lang_pairs

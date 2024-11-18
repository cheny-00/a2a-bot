# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import torch
import importlib
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, default_collate
from torch.utils.data.sampler import WeightedRandomSampler

from utils.utils import get_mapped_kwargs


class DataInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8, dataset="", batch_size=4, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.test_batch_size = kwargs.get("val_batch_size", self.batch_size)
        self.data_module = None
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = self.data_module(
                data_dir=self.kwargs["train_data_dir"],
                **get_mapped_kwargs(self.data_module.__init__, self.kwargs),
                train=True,
            )
            if self.kwargs["valid_data_dir"] == "random":
                indices = list(range(len(self.train_set)))
                random.shuffle(indices)
                split_rate = int(0.9 * len(self.train_set)) 
                train_indices = indices[:split_rate]
                val_indices = indices[split_rate:]
                self.val_set = Subset(self.train_set, val_indices)
                self.train_set = Subset(self.train_set, train_indices)
            else:
                self.val_set = self.data_module(
                    data_dir=self.kwargs["valid_data_dir"],
                    **get_mapped_kwargs(self.data_module.__init__, self.kwargs),
                    train=False,
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = self.data_module(
                data_dir=self.kwargs["test_data_dir"],
                **get_mapped_kwargs(self.data_module.__init__, self.kwargs),
                train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except Exception as e:
            print(f"Error message: {e}")
            raise ValueError(
                f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!"
            )


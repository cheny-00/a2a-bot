# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import torch
import importlib
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, default_collate, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

from utils.utils import get_mapped_kwargs


class DataInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8, datasets=None, batch_size=4, sampling_weights=None, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        # Convert single dataset to list for compatibility
        print(f"========= Use datasets: {datasets} =========")
        self.datasets = [datasets] if isinstance(datasets, str) else datasets
        assert "train_data_dir" in kwargs, "train_data_dir is required"
        train_data_dir = kwargs["train_data_dir"]
        train_data_dir_list = train_data_dir.split(",") if type(train_data_dir) == str else train_data_dir
        if len(train_data_dir_list) == 1:
            train_data_dir_list = [train_data_dir_list[0]] * len(self.datasets)
        assert len(train_data_dir_list) == len(self.datasets), "The number of train_data_dirs must be equal to the number of datasets, or train_data_dir must be a single directory"
        
        self.valid_data_dir = kwargs.get("valid_data_dir", "random")
        
        self.sampling_weights = sampling_weights
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.test_batch_size = kwargs.get("val_batch_size", self.batch_size)
        self.train_data_dir_map = dict()
        self.data_modules = []
        self.load_data_modules()

    def load_data_modules(self):
        for name in self.datasets:
            camel_name = "".join([i.capitalize() for i in name.split("_")])
            try:
                module = getattr(
                    importlib.import_module("." + name, package=__package__), camel_name
                )
                self.data_modules.append(module)
                self.train_data_dir_map[module.__name__] = self.kwargs["train_data_dir"]
                
            except Exception as e:
                print(f"Error message: {e}")
                raise ValueError(
                    f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!"
                )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Create a list to store all training datasets
            self.train_sets = []
            self.val_sets = []
            
            for data_module in self.data_modules:
                print(f"========= Use train_data_dir: {self.train_data_dir_map[data_module.__name__]} for dataset: {data_module.__name__} =========")
                train_set = data_module(
                    data_dir=self.train_data_dir_map[data_module.__name__],
                    **get_mapped_kwargs(data_module.__init__, self.kwargs),
                    train=True,
                )
                
                if self.kwargs["valid_data_dir"] == "random":
                    indices = list(range(len(train_set)))
                    random.shuffle(indices)
                    split_rate = int(0.9 * len(train_set))
                    train_indices = indices[:split_rate]
                    val_indices = indices[split_rate:]
                    self.val_sets.append(Subset(train_set, val_indices))
                    self.train_sets.append(Subset(train_set, train_indices))
                else:
                    val_set = data_module(
                        data_dir=self.kwargs["valid_data_dir"],
                        **get_mapped_kwargs(data_module.__init__, self.kwargs),
                        train=False,
                    )
                    self.train_sets.append(train_set)
                    self.val_sets.append(val_set)

            # Combine datasets using ConcatDataset
            self.train_set = ConcatDataset(self.train_sets)
            self.val_set = ConcatDataset(self.val_sets)

        if stage == "test" or stage is None:
            self.test_sets = []
            for data_module in self.data_modules:
                test_set = data_module(
                    data_dir=self.kwargs["test_data_dir"],
                    **get_mapped_kwargs(data_module.__init__, self.kwargs),
                    train=False,
                )
                self.test_sets.append(test_set)
            self.test_set = ConcatDataset(self.test_sets)

    def train_dataloader(self):
        # Calculate sampling weights if not provided
        if self.sampling_weights is None and len(self.sampling_weights) == len(self.train_sets):
            # Equal probability for each dataset
            weights = [1.0/len(self.train_sets)] * len(self.train_set)
        else:
            # Use provided sampling weights
            total_weight = sum(self.sampling_weights)
            weights = [w/total_weight for w in self.sampling_weights]

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.train_set),
            replacement=True
        )

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
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


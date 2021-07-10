from os import PathLike
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

IntoPath = Union[Path, PathLike, str]


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        self.lr = torch.from_numpy(np.load(name.with_suffix(".lr.npy"), mmap_mode="c"))
        self.hr = torch.from_numpy(np.load(name.with_suffix(".hr.npy"), mmap_mode="c"))

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        lr = self.lr[index]
        hr = self.hr[index]

        return lr, hr

    def __len__(self):
        return len(self.lr)


class TensorPairsDataModule(LightningDataModule):
    train_with: IntoPath
    test_with: IntoPath

    batch_size: int

    training: TensorPairsDataset
    testing: TensorPairsDataset

    training_indices: Any
    validation_indices: Any

    def __init__(
        self,
        train_with: IntoPath,
        test_with: IntoPath,
        batch_size: int,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.test_with = test_with
        self.train_with = train_with

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.training = TensorPairsDataset(self.train_with)

            # Splits are fully deterministic across everything.
            self.training_indices, self.validation_indices = next(
                KFold(n_splits=5, shuffle=False).split(self.training)
            )

        if stage in (None, "test"):
            self.testing = TensorPairsDataset(self.test_with)

    def train_dataloader(self):
        return DataLoader(
            self.training,
            self.batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=SubsetRandomSampler(self.training_indices),
        )

    def val_dataloader(self):
        return DataLoader(
            self.training,
            self.batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=SubsetRandomSampler(self.validation_indices),
        )

    def test_dataloader(self):
        return DataLoader(
            self.testing,
            self.batch_size,
            drop_last=True,
            pin_memory=True,
        )

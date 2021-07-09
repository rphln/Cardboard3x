from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

IntoPath = Union[Path, PathLike, str]


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        with h5py.File(name, "r") as h5:
            self.lr = torch.as_tensor(h5["lr"][:])
            self.hr = torch.as_tensor(h5["hr"][:])

    def __getitem__(self, index: int):
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
    validation: TensorPairsDataset
    testing: TensorPairsDataset

    validation_ratio: float = 0.2

    def __init__(
        self,
        train_with: IntoPath,
        test_with: IntoPath,
        batch_size: int,
        validation_ratio: float = validation_ratio,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.test_with = test_with
        self.train_with = train_with

        self.validation_ratio = validation_ratio

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.training, self.validation = train_test_split(
                TensorPairsDataset(self.train_with),
                test_size=self.validation_ratio,
                shuffle=False,
            )

        if stage in (None, "test"):
            self.testing = TensorPairsDataset(self.test_with)

    def train_dataloader(self):
        return DataLoader(
            self.training,
            self.batch_size,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            self.batch_size,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testing,
            self.batch_size,
            drop_last=True,
            pin_memory=True,
        )

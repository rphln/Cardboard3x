from typing import Tuple

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        self.h5 = h5py.File(name, "r", libver="latest", swmr=True)

    def __getitem__(self, index: int):
        lr = torch.from_numpy(self.h5["lr"][index])
        hr = torch.from_numpy(self.h5["hr"][index])

        return lr, hr

    def __len__(self):
        return len(self.h5["lr"])


__all__ = ("TensorPairsDataset",)

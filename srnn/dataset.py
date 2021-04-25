from typing import Tuple

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        h5 = h5py.File(name, "r", libver="latest", swmr=True)

        self.lr = torch.from_numpy(h5["lr"][:])
        self.hr = torch.from_numpy(h5["hr"][:])

    def __getitem__(self, index: int):
        return self.lr[index], self.hr[index]

    def __len__(self):
        return len(self.lr)


__all__ = ("TensorPairsDataset",)

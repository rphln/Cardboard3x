from pathlib import Path
from typing import Tuple

import torch
from numpy.lib.format import open_memmap
from torch import Tensor
from torch.utils.data import Dataset


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, path: Path):
        lr = path.with_suffix(".lr.npy")
        hr = path.with_suffix(".hr.npy")

        self.lr = torch.from_numpy(open_memmap(lr, mode="c"))
        self.hr = torch.from_numpy(open_memmap(hr, mode="c"))

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        lr = self.lr[index]
        hr = self.hr[index]

        return lr, hr

    def __len__(self):
        return len(self.lr)

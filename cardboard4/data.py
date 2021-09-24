from pathlib import Path
from typing import Tuple

import torch
from numpy.lib.format import open_memmap
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, path: Path):
        lr = path.with_suffix(".lr.npy")
        hr = path.with_suffix(".hr.npy")

        self.lr = torch.from_numpy(open_memmap(lr, mode="c"))
        self.hr = torch.from_numpy(open_memmap(hr, mode="c"))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = self.lr[index]
        hr = self.hr[index]

        return lr, hr

    def __len__(self):
        return len(self.lr)


class FilePairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, path: Path):
        super().__init__()

        self.lr = sorted(path.glob("**/*.LR.png"))
        self.hr = sorted(path.glob("**/*.HR.png"))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = to_tensor(default_loader(self.lr[index]))
        hr = to_tensor(default_loader(self.hr[index]))

        return lr, hr

    def __len__(self):
        return len(self.lr)

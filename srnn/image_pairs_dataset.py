from functools import lru_cache
from os import PathLike
from pathlib import Path
from random import choice
from sys import stderr
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import (
    InterpolationMode,
    gaussian_blur,
    resize,
    to_tensor,
)


class ImagePairsDataset(Dataset):
    extensions = {".png", ".jpg"}

    def __init__(self, path: PathLike, lr_size: int, scale: int):
        self.files = [
            file for file in Path(path).glob("**/*") if file.suffix in self.extensions
        ]

        self.lr_size = lr_size
        self.hr_size = lr_size * scale

        self.crop = RandomCrop(size=self.hr_size)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        path: Path = self.files[index]

        try:
            file = default_loader(path)
        except Exception as err:
            print(f"Warning: Invalid file found at `{path}`", file=stderr)
            raise err

        hr: Tensor = to_tensor(self.crop(file))
        lr: Tensor = resize(
            gaussian_blur(hr, kernel_size=3), self.lr_size, InterpolationMode.BICUBIC
        )

        return lr, hr

    def __len__(self):
        return len(self.files)

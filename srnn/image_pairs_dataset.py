from functools import lru_cache
from os import PathLike
from pathlib import Path
from random import choice
from typing import Sequence, Tuple

from PIL import Image
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
    def __init__(self, path: PathLike):
        self.groups = [group for group in Path(path).glob("*") if group.is_dir()]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        files = list(self.groups[index].glob("*.hr"))

        hr: Path = choice(files)
        lr: Path = hr.with_suffix(".lr")

        return to_tensor(default_loader(lr)), to_tensor(default_loader(hr))

    def __len__(self):
        return len(self.groups)

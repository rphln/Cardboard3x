from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import (
    InterpolationMode,
    gaussian_blur,
    resize,
    to_tensor,
)


class ImagePairsDataset(Dataset):
    def __init__(self, files: Sequence[Path], lr_size: int, scale: int):
        self.files = list(files)
        self.lr_size = lr_size

        # This is a stateful transformation, and as such needs to be maintained
        # across each access.
        self.crop = RandomCrop(size=lr_size * scale)

    @lru_cache(maxsize=None)
    def __load_image(self, index):
        return to_tensor(Image.open(self.files[index]).convert("YCbCr").getchannel("Y"))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # y = to_tensor(Image.open(self.files[index]).convert("YCbCr").getchannel("Y"))
        y = self.__load_image(index)

        hr = self.crop(y)
        lr = resize(
            gaussian_blur(hr, kernel_size=3), self.lr_size, InterpolationMode.BICUBIC
        )

        return lr, hr

    def __len__(self):
        return len(self.files)

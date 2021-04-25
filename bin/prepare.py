#!/usr/bin/env -S poetry run python

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Union

import click
import h5py
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.functional import Tensor
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import (
    center_crop,
    gaussian_blur,
    resize,
    to_tensor,
)
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def scale(image: Tensor, size: Union[int, Tuple[int, int]]) -> Tensor:
    """
    Applies a single pyramid scale step to an image.
    """

    return resize(gaussian_blur(image, kernel_size=3), size)


def pyramid(image: Tensor, target_size: int) -> Tensor:
    _, height, width = image.shape

    half_width = width // 2
    half_height = height // 2

    if half_width < target_size or half_height < target_size:
        return image

    return pyramid(scale(image, (half_height, half_width)), target_size)


def process(source: Path, lr_size: int, hr_size: int) -> Tuple[Tensor, Tensor]:
    image: Tensor = pyramid(to_tensor(default_loader(source)), hr_size)

    hr = center_crop(image, hr_size)
    lr = scale(hr, lr_size)

    return lr, hr


def convert(
    source: List[Path], target: Path, lr_size: int, scale_factor: int, channels: int = 3
):
    with h5py.File(target, "w", libver="latest") as file:
        with Pool() as pool:
            hr_size = lr_size * scale_factor

            lr_dataset = file.create_dataset(
                "lr", dtype="f", shape=(len(source), channels, lr_size, lr_size)
            )
            hr_dataset = file.create_dataset(
                "hr", dtype="f", shape=(len(source), channels, hr_size, hr_size)
            )

            process_ = partial(process, lr_size=lr_size, hr_size=hr_size)

            for idx, (lr, hr) in enumerate(
                pool.imap_unordered(process_, tqdm(source, unit="file"))
            ):
                lr_dataset[idx] = lr.numpy()
                hr_dataset[idx] = hr.numpy()


def prepare(folder: Path, lr_size: int, scale_factor: int):
    files = []
    files += folder.glob("**/*.png")
    files += folder.glob("**/*.jpg")

    train, test = train_test_split(files, test_size=0.2)

    convert(test, folder.with_suffix(".test.h5"), lr_size, scale_factor)
    convert(train, folder.with_suffix(".train.h5"), lr_size, scale_factor)


@click.command()
@click.option("--folder", required=True, type=Path)
@click.option("--lr-size", required=True, type=int)
@click.option("--scale-factor", required=True, type=int)
def cli(folder, lr_size, scale_factor):
    return prepare(folder, lr_size, scale_factor)


if __name__ == "__main__":
    cli()

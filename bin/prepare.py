#!/usr/bin/env -S poetry run python

from pathlib import Path

import click
from torch.functional import Tensor
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import (
    InterpolationMode,
    gaussian_blur,
    resize,
    to_tensor,
)
from torchvision.utils import save_image
from tqdm import tqdm


def to_patches(image: Tensor, size: int) -> Tensor:
    """
    Converts an image to a batch of non-overlapping tiles of the specified `size`.
    """

    # Splits a dimension in non-overlapping tiles of the specified size.
    image = image.unfold(dimension=0, size=3, step=3)
    image = image.unfold(dimension=1, size=size, step=size)
    image = image.unfold(dimension=2, size=size, step=size)

    # Turns the tiles in a batch.
    return image.reshape(-1, 3, size, size)


def prepare(source: Path, target: Path, lr_size: int, scale_factor: int, glob: str):
    hr_size = lr_size * scale_factor

    files = list(source.glob(glob))

    for image in tqdm(files, unit="image"):
        hr = to_patches(
            to_tensor(default_loader(image)),
            hr_size,
        )
        lr = resize(
            gaussian_blur(hr, kernel_size=3),
            lr_size,
            InterpolationMode.BICUBIC,
        )

        for idx, (x, y) in enumerate(zip(lr, hr)):
            group = target / image.name
            group.mkdir(exist_ok=True, parents=True)

            save_image(x, group / f"{idx}.lr", format="png")
            save_image(y, group / f"{idx}.hr", format="png")


@click.command()
@click.option("--source", required=True, type=Path)
@click.option("--target", required=True, type=Path)
@click.option("--lr-size", required=True, type=int)
@click.option("--scale-factor", required=True, type=int)
@click.option("--glob", default="*.png")
def cli(source, target, lr_size, scale_factor, glob):
    return prepare(source, target, lr_size, scale_factor, glob)


if __name__ == "__main__":
    cli()

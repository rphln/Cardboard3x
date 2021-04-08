#!/usr/bin/env -S poetry run python

from pathlib import Path

import click
import h5py
from PIL import Image
from torch.nn.functional import unfold
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import (
    InterpolationMode,
    center_crop,
    gaussian_blur,
    resize,
    to_tensor,
)
from tqdm import tqdm


def prepare(folder: Path, lr_size: int, scale_factor: int, glob: str):
    files = list(folder.glob(glob))
    count = len(files)

    target = folder.with_suffix(".d")
    target.mkdir(parents=True, exist_ok=True)

    with h5py.File(folder.with_suffix(".h5"), "w", libver="latest") as f:
        lr = f.create_group("lr")
        hr = f.create_group("hr")

        for idx, image in enumerate(tqdm(files, unit="image")):
            name = str(idx)

            # y = to_tensor(Image.open(image).convert("YCbCr").getchannel("Y"))
            y = to_tensor(default_loader(image))

            y = y.unfold(1, 96, 96)
            y = y.unfold(2, 96, 96)
            y = y.reshape(-1, 3, 96, 96)
            x = resize(
                gaussian_blur(y, kernel_size=3), lr_size, InterpolationMode.BICUBIC
            )

            lr.create_dataset(name, chunks=True, compression="gzip", data=x)
            hr.create_dataset(name, chunks=True, compression="gzip", data=y)

        # for tile, (lr_, hr_) in enumerate(zip(lr, hr)):
        #     target_ = target / str(idx)
        #     target_.mkdir(exist_ok=True, parents=True)

        #     save_image(lr_, target_ / f"{tile}_lr.png")
        #     save_image(hr_, target_ / f"{tile}_hr.png")

        # t = to_tensor(y)
        # t = center_crop(t, 512, 512)
        # t = t.unfold(1, 96, 96)
        # t = t.unfold(2, 96, 96)
        # t = t.reshape(-1, channels, 96, 96)
        # save_image(make_grid(t), "/home/rphln/test.png")

        # hr = center_crop(y, hr_size)
        # lr = resize(
        #     gaussian_blur(hr, kernel_size=3), lr_size, InterpolationMode.BICUBIC
        # )

        # lr_dataset[i] = to_tensor(lr).numpy()
        # hr_dataset[i] = to_tensor(hr).numpy()

    # with h5py.File(folder.with_suffix(".h5"), "w", libver="latest") as f:
    #     hr_size = lr_size * scale_factor

    #     lr_dataset = f.create_dataset(
    #         "lr",
    #         dtype="f",
    #         shape=(count, 1, lr_size, lr_size),
    #         chunks=True,
    #         compression="gzip",
    #     )
    #     hr_dataset = f.create_dataset(
    #         "hr",
    #         dtype="f",
    #         shape=(count, 1, hr_size, hr_size),
    #         chunks=True,
    #         compression="gzip",
    #     )

    #     for i, image in enumerate(tqdm(files, unit="image")):
    #         y = Image.open(image).convert("YCbCr").getchannel("Y")

    #         hr = center_crop(y, hr_size)
    #         lr = resize(
    #             gaussian_blur(hr, kernel_size=3), lr_size, InterpolationMode.BICUBIC
    #         )

    #         lr_dataset[i] = to_tensor(lr).numpy()
    #         hr_dataset[i] = to_tensor(hr).numpy()


@click.command()
@click.option("--folder", required=True, type=Path)
@click.option("--lr-size", required=True, type=int)
@click.option("--scale-factor", required=True, type=int)
@click.option("--glob", default="*.png")
def cli(folder, lr_size, scale_factor, glob):
    return prepare(folder, lr_size, scale_factor, glob)


if __name__ == "__main__":
    cli()

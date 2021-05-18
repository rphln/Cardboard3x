#!/usr/bin/env -S poetry run python

from pathlib import Path

import click
import torch
from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from PIL import Image
from torch.nn import Conv2d, Module
from torch.nn.functional import interpolate, relu
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image


def conv2d(in_channels, out_channels, kernel_size) -> Module:
    return Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


class SRCNN(Module):
    N0 = 1
    N1 = 64
    N2 = 32

    F1 = 9
    F2 = 5
    F3 = 5

    def __init__(self):
        super().__init__()

        self.conv1 = conv2d(
            in_channels=self.N0, out_channels=self.N1, kernel_size=self.F1
        )
        self.conv2 = conv2d(
            in_channels=self.N1, out_channels=self.N2, kernel_size=self.F2
        )
        self.conv3 = conv2d(
            in_channels=self.N2, out_channels=self.N0, kernel_size=self.F3
        )

    def forward(self, x):
        x = rgb_to_ycbcr(x)

        x = interpolate(x, scale_factor=3, mode="bicubic", align_corners=False)

        # Extract the Y channel.
        y = x[:, :1, :, :]

        y = relu(self.conv1(y), inplace=True)
        y = relu(self.conv2(y), inplace=True)
        y = self.conv3(y)

        # Overwrite the Y channel in the partial conversion.
        x[:, :1, :, :] = y

        return ycbcr_to_rgb(x)


@torch.no_grad()
def upscale(device, state, source, target):
    model = SRCNN().to(device)
    state = torch.load(state)

    model.load_state_dict(state["model_state_dict"])

    lr = to_tensor(Image.open(source).convert("RGB")).unsqueeze_(0)
    save_image(model(lr), target, format="png")


@click.command()
@click.option("--device", default="cpu", type=torch.device)
@click.option("--state", required=True, type=Path)
@click.argument("source", type=Path)
@click.argument("target", type=Path)
def cli(device, state, source, target):
    return upscale(device, state, source, target)


if __name__ == "__main__":
    cli()

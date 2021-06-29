#!/usr/bin/env -S poetry run python

from pathlib import Path

import click
import torch
from PIL import Image
from torch.nn import Conv2d, Module, PixelShuffle, Sequential, Tanh
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image


def conv2d(in_channels, out_channels, kernel_size) -> Module:
    return Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def init_parameters(module):
    if isinstance(module, Conv2d):
        init.xavier_normal_(module.weight)
        init.constant_(module.bias, 0.0)


class ESPCN(Module):
    SCALE = 3

    N0 = 3
    N1 = 64
    N2 = 32

    F1 = 5
    F2 = 3
    F3 = 3

    def __init__(self):
        super().__init__()

        self.conv1 = conv2d(
            in_channels=self.N0, out_channels=self.N1, kernel_size=self.F1
        )
        self.conv2 = conv2d(
            in_channels=self.N1, out_channels=self.N2, kernel_size=self.F2
        )
        self.conv3 = conv2d(
            in_channels=self.N2,
            out_channels=self.N0 * self.SCALE ** 2,
            kernel_size=self.F3,
        )

        self.sequential = Sequential(
            self.conv1, Tanh(), self.conv2, Tanh(), self.conv3, PixelShuffle(self.SCALE)
        )

    def forward(self, x):
        return self.sequential(x)


@torch.no_grad()
def upscale(device, state, source, target):
    model = ESPCN().to(device)
    state = torch.load(state)

    model.load_state_dict(state["model_state_dict"])

    lr = to_tensor(Image.open(source).convert("RGB")).unsqueeze_(0)
    hr = model(lr).clamp(0.0, 1.0)

    save_image(hr, target, format="png")


@click.command()
@click.option("--device", default="cpu", type=torch.device)
@click.option("--state", required=True, type=Path)
@click.argument("source", type=Path)
@click.argument("target", type=Path)
def cli(device, state, source, target):
    return upscale(device, state, source, target)


if __name__ == "__main__":
    cli()

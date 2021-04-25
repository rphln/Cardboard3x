#!/usr/bin/env -S poetry run python

from pathlib import Path

import click
import torch
from kornia.color import ycbcr_to_rgb
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, resize, to_tensor
from torchvision.utils import save_image

from srnn.models.srcnn import SRCNN


@torch.no_grad()
def upscale(device, state, source, target):
    model = SRCNN(channels=1).to(device)
    state = torch.load(state)

    model.load_state_dict(state["model_state_dict"])

    lr = to_tensor(Image.open(source).convert("YCbCr"))
    lr = lr.unsqueeze_(0)

    _, _, height, width = lr.shape

    hr = resize(lr, (height * 3, width * 3), InterpolationMode.BICUBIC)

    # Forward the luminance channel.
    hr[:, :1, :, :] = model(lr[:, :1, :, :]).clamp(0.0, 1.0)

    save_image(ycbcr_to_rgb(hr), target, format="png")


@click.command()
@click.option("--device", default="cpu", type=torch.device)
@click.option("--state", required=True, type=Path)
@click.argument("source", type=Path)
@click.argument("target", type=Path)
def cli(device, state, source, target):
    return upscale(device, state, source, target)


if __name__ == "__main__":
    cli()

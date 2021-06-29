#!/usr/bin/env -S poetry run python3

from pathlib import Path
from typing import Tuple

import click
import h5py
import torch
from kornia.losses import psnr, ssim
from torch import Tensor
from torch.nn import Upsample
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset


# %%
class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        h5 = h5py.File(name, "r", libver="latest", swmr=True)

        self.lr = torch.from_numpy(h5["lr"][:])
        self.hr = torch.from_numpy(h5["hr"][:])

    def __getitem__(self, index: int):
        return self.lr[index], self.hr[index]

    def __len__(self):
        return len(self.lr)


# %%
@torch.no_grad()
def forward(device, dataset, batch_size):
    model = Upsample(scale_factor=3, mode="bicubic").to(device)

    loader = DataLoader(dataset, batch_size, drop_last=True, shuffle=False)

    model.eval()

    mean_psnr = 0.0
    mean_ssim = 0.0

    for batch, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        z = model(x).clamp(0.0, 1.0)

        mean_psnr += (psnr(z, y, max_val=1.0).mean().item() - mean_psnr) / batch
        mean_ssim += (ssim(z, y, window_size=5).mean().item() - mean_ssim) / batch

    return mean_psnr, mean_ssim


@click.command()
@click.option("--device", default="cuda:0", type=torch.device)
@click.option("--models", required=True, type=Path)
@click.option("--dataset", required=True, type=Path)
@click.option("--batch-size", default=32)
def cli(device, models, dataset, batch_size):
    dataset = TensorPairsDataset(dataset)
    print("-", *forward(device, dataset, batch_size), sep="\t")


if __name__ == "__main__":
    cli()

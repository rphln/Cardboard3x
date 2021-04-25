#!/usr/bin/env -S poetry run python3

from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from srnn import stats
from srnn.dataset import PairsDataset
from srnn.models.srcnn import SRCNN


@torch.no_grad()
def forward(device, state, dataset, batch_size):
    model = SRCNN(channels=1).to(device)
    criterion = torch.nn.MSELoss()

    state = torch.load(state)
    model.load_state_dict(state["model_state_dict"])

    loader = DataLoader(PairsDataset(dataset), batch_size)

    loss = 0.0
    psnr = 0.0
    ssim = 0.0

    for batch, (x, y) in enumerate(tqdm(loader), start=1):
        y = y.to(device)
        x = x.to(device)

        z = model(x).clamp(0.0, 1.0)

        results = stats(None, "Forward", criterion, y, z)

        loss += (results["Loss"] - loss) / batch
        psnr += (results["PSNR"] - psnr) / batch
        ssim += (results["SSIM"] - ssim) / batch

    print(f"Loss: {loss:.5f}")
    print(f"PSNR: {psnr:.5f}")
    print(f"SSIM: {ssim:.5f}")


@click.command()
@click.option("--device", default="cpu", type=torch.device)
@click.option("--state", required=True, type=Path)
@click.option("--dataset", required=True, type=Path)
@click.option("--batch-size", default=32)
def cli(device, state, dataset, batch_size):
    return forward(device, state, dataset, batch_size)


if __name__ == "__main__":
    cli()

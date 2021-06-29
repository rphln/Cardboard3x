#!/usr/bin/env -S poetry run python3

from pathlib import Path
from typing import Tuple

import click
import h5py
import torch
from kornia.losses import psnr, ssim
from torch import Tensor
from torch.nn import Conv2d, Module, PixelShuffle, Sequential, Tanh
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from tqdm.auto import tqdm

from torch.nn import Conv2d, Module, init
from torch.nn.functional import interpolate, mse_loss, relu
from torchsummary import summary
from torchvision.transforms.functional import normalize
from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from torch.nn import Conv2d, Module, init
from torch.nn.functional import interpolate, l1_loss, leaky_relu, mse_loss, relu
from torchvision.transforms.functional import normalize

# %%

def conv2d(in_channels, out_channels, kernel_size) -> Module:
    return Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def init_parameters(module):
    if isinstance(module, Conv2d):
        init.normal_(module.weight, std=1e-3)
        init.constant_(module.bias, val=0e-0)


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
def forward(device, state, dataset, batch_size):
    model = SRCNN().to(device)

    state = torch.load(state)
    model.load_state_dict(state["model_state_dict"])

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

    for state in sorted(models.glob("*.pth")):
        print(state.stem, *forward(device, state, dataset, batch_size), sep="\t")


if __name__ == "__main__":
    cli()

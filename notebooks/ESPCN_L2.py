# %%
from itertools import count
from math import inf
from pathlib import Path
from typing import Tuple

import h5py
import torch
from google.colab import drive
from sklearn.model_selection import KFold
from torch import Tensor
from torch.nn import Conv2d, Module, MSELoss, PixelShuffle, Sequential, Tanh, init
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from tqdm.auto import tqdm

# %%
drive.mount("/content/drive")


# %%
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
    N3 = N0 * SCALE ** 2

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
            in_channels=self.N2, out_channels=self.N3, kernel_size=self.F3
        )

        self.head = Sequential(self.conv1, Tanh(), self.conv2, Tanh())
        self.tail = Sequential(self.conv3, PixelShuffle(self.SCALE))

        self.sequential = Sequential(self.head, self.tail)

    def forward(self, x):
        return self.sequential(x)


# %%
class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        h5 = h5py.File(name, "r", libver="latest", swmr=True)

        self.lr = h5["lr"]
        self.hr = h5["hr"]

    def __getitem__(self, index: int):
        return torch.from_numpy(self.lr[index]), torch.from_numpy(self.hr[index])

    def __len__(self):
        return len(self.lr)


# %%


def train(
    model: Module,
    criterion: Module,
    device: Device,
    optimizer: Optimizer,
    loader: DataLoader,
) -> float:
    """
    Runs a training epoch.
    """

    model.train()
    epoch_mean_loss = 0

    for batch, (x, y) in enumerate(
        tqdm(loader, unit="batch", leave=False, desc="Training"), start=1
    ):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        loss = criterion(model(x), y)
        epoch_mean_loss += (loss.item() - epoch_mean_loss) / batch

        loss.backward()
        optimizer.step()

    return epoch_mean_loss


@torch.no_grad()
def validate(
    model: Module,
    criterion: Module,
    device: Device,
    loader: DataLoader,
) -> float:
    """
    Validates the model.
    """

    model.eval()
    epoch_mean_loss = 0

    for batch, (x, y) in enumerate(
        tqdm(loader, unit="batch", leave=False, desc="Validation"), start=1
    ):
        x = x.to(device)
        y = y.to(device)

        loss = criterion(model(x), y)
        epoch_mean_loss += (loss.item() - epoch_mean_loss) / batch

    return epoch_mean_loss


# %%
LEARNING_RATE = 1e-4
BATCH_SIZE = 1024

EPOCHS = 300
PATIENCE = 10

checkpoints = Path("/content/drive/MyDrive/Checkpoints/") / "ESPCN (L2, Scheduled)"
checkpoints.mkdir(parents=True, exist_ok=True)

resume = None
resume = checkpoints / "0.pth"

# dataset = TensorPairsDataset("var/rphln-safebooru2020-medium.train.h5")
dataset = TensorPairsDataset(
    "/content/drive/MyDrive/Datasets/rphln-safebooru2020-medium.train.h5"
)

criterion = MSELoss()

device = torch.device("cuda:0")
summary(ESPCN().to(device), input_size=(3, 32, 32))

# %%
checkpoint = torch.load(resume) if resume else {}

folds = KFold(n_splits=5, shuffle=False).split(dataset)

for fold, (training_indices, validation_indices) in enumerate(folds):
    if checkpoint and fold < checkpoint["fold"]:
        continue

    checkpoints_ = checkpoints / str(fold)
    checkpoints_.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir=str(checkpoints_))

    model = ESPCN().to(device)
    model.apply(init_parameters)

    parameters = [
        {"params": model.head.parameters()},
        {"params": model.tail.parameters(), "lr": LEARNING_RATE * 0.1},
    ]

    optimizer = Adam(parameters, lr=LEARNING_RATE)

    if checkpoint:
        start = checkpoint["epoch"]
        best = checkpoint["best"]

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint = {}
    else:
        start = 0
        best = inf

    training_loader = DataLoader(
        dataset,
        BATCH_SIZE,
        drop_last=True,
        pin_memory=True,
        sampler=SubsetRandomSampler(training_indices),
    )
    validation_loader = DataLoader(
        dataset,
        BATCH_SIZE,
        drop_last=True,
        pin_memory=True,
        sampler=SubsetRandomSampler(validation_indices),
    )

    # How many epochs since the last improvement?
    stale = 0

    for epoch in tqdm(count(start=1 + start), unit="epoch", desc=f"Fold #{fold}"):
        epoch_training_loss = train(
            model,
            criterion,
            device,
            optimizer,
            training_loader,
        )
        epoch_validation_loss = validate(
            model,
            criterion,
            device,
            validation_loader,
        )

        if epoch_validation_loss < best:
            best = epoch_validation_loss
            stale = 0

            stats = {
                "Training": epoch_training_loss,
                "Validation": epoch_validation_loss,
            }
            writer.add_scalars("Loss", stats, epoch)

            state = {
                "name": model.__class__.__name__,
                "epoch": epoch,
                "fold": fold,
                "best": best,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(state, checkpoints_.with_suffix(".pth"))
        else:
            stale += 1

        if stale >= PATIENCE:
            break

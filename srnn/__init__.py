from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import KFold
from torch.nn import Module
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm, trange


class Model(Module):
    @property
    def name(self) -> str:
        """
        Returns a name to identify the model files.
        """

        raise NotImplementedError


def setup_drive():
    try:
        from google.colab import drive

        drive.mount("/content/drive/", force_remount=True)
    except ModuleNotFoundError:
        pass


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
    model: Module, criterion: Module, device: Device, loader: DataLoader
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


def training(
    model: Model,
    parameters: Any,
    criterion: Module,
    epochs: int,
    batch_size: int,
    save_interval: int,
    device: Device,
    dataset: Dataset,
    checkpoints: Path,
    name: str,
    learning_rate: float,
):
    checkpoints = checkpoints / datetime.now().isoformat()
    checkpoints.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = Adam(parameters, lr=learning_rate)

    folding = KFold(n_splits=5)

    for fold, (training_indices, validation_indices) in enumerate(
        folding.split(dataset)
    ):
        group = name.format(model=model.name, fold=fold, epoch=0)
        writer = SummaryWriter(log_dir=(checkpoints / group).with_suffix(""))

        training_loader = DataLoader(
            dataset,
            batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=SubsetRandomSampler(training_indices),
        )
        validation_loader = DataLoader(
            dataset,
            batch_size,
            drop_last=True,
            pin_memory=True,
            sampler=SubsetRandomSampler(validation_indices),
        )

        for epoch in trange(0, epochs, unit="epoch", desc=f"Fold #{fold}"):
            training_loss = train(model, criterion, device, optimizer, training_loader)
            validation_loss = validate(model, criterion, device, validation_loader)

            writer.add_scalar("Loss/Training", training_loss, epoch)
            writer.add_scalar("Loss/Validation", validation_loss, epoch)

            if epoch % save_interval == 0:
                checkpoint = name.format(model=model.name, epoch=epoch, fold=fold)

                state = {
                    "name": checkpoint,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                torch.save(state, checkpoints / checkpoint)

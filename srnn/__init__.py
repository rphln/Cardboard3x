from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from kornia.losses import psnr, ssim
from sklearn.model_selection import KFold
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange


def reset_parameters(module):
    if callable(getattr(module, "reset_parameters", None)):
        module.reset_parameters()


def mean_psnr(x: Tensor, y: Tensor) -> Tensor:
    return psnr(x, y, max_val=1.0).mean()


def mean_ssim(x: Tensor, y: Tensor) -> Tensor:
    return ssim(x, y, window_size=3).mean()


def to_name(base, model, epoch, fold):
    return base.format(model=model.__class__.__name__, epoch=epoch, fold=fold)


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
        z = model(x)

        loss = criterion(z, y)
        epoch_mean_loss += (loss.item() - epoch_mean_loss) / batch

        loss.backward()
        optimizer.step()

    return epoch_mean_loss


@torch.no_grad()
def validate(model: Module, device: Device, loader: DataLoader) -> Tuple[float, float]:
    """
    Validates the model.
    """

    model.eval()

    epoch_mean_psnr = 0
    epoch_mean_ssim = 0

    for batch, (x, y) in enumerate(
        tqdm(loader, unit="batch", leave=False, desc="Validation"), start=1
    ):
        x = x.to(device)
        y = y.to(device)
        z = model(x)

        epoch_mean_psnr += (mean_psnr(z, y).item() - epoch_mean_psnr) / batch
        epoch_mean_ssim += (mean_ssim(z, y).item() - epoch_mean_ssim) / batch

    return epoch_mean_psnr, epoch_mean_ssim


def training(
    model: Module,
    parameters: Any,
    criterion: Module,
    epochs: int,
    batch_size: int,
    save_interval: int,
    device: Device,
    dataset: Dataset,
    checkpoints: Path,
    basename: str,
    learning_rate: float,
    resume: Optional[Path] = None,
    init_parameters: Callable[[Module], None] = lambda _: None,
):
    checkpoints.mkdir(parents=True, exist_ok=True)

    if resume:
        checkpoint: Dict = torch.load(resume)
    else:
        checkpoint = {}

    folds = KFold(n_splits=5, shuffle=False).split(dataset)

    for fold, (training_indices, validation_indices) in islice(
        enumerate(folds), checkpoint.get("fold", 0), None
    ):
        model = model.to(device)

        model.apply(reset_parameters)
        model.apply(init_parameters)

        optimizer = Adam(parameters, lr=learning_rate)

        if checkpoint:
            start = checkpoint["epoch"]

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            checkpoint = {}
        else:
            start = 0

        log_dir = (checkpoints / to_name(basename, model, epochs, fold)).with_suffix("")
        writer = SummaryWriter(log_dir=log_dir)

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

        for epoch in trange(1 + start, 1 + epochs, unit="epoch", desc=f"Fold #{fold}"):
            loss = train(model, criterion, device, optimizer, training_loader)

            if epoch % save_interval == 0:
                psnr_, ssim_ = validate(model, device, validation_loader)

                writer.add_scalar("PSNR/Validation", psnr_, epoch)
                writer.add_scalar("SSIM/Validation", ssim_, epoch)

                writer.add_scalar("Loss/Training", loss, epoch)

                state = {
                    "name": model.__class__.__name__,
                    "epoch": epoch,
                    "fold": fold,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                torch.save(state, checkpoints / to_name(basename, model, epoch, fold))

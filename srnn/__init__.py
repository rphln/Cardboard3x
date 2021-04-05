from pathlib import Path

import torch
from kornia.losses import psnr, ssim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import trange

try:
    from srnn.dataset import PairsDataset
    from srnn.models.srcnn import SRCNN
except ModuleNotFoundError:
    # Safe to ignore on Colab.
    pass


def stats(epoch, type, criterion, y, z):
    with torch.no_grad():
        return {
            "Epoch": epoch,
            "Type": type,
            "PSNR": psnr(y, z, max_val=1.0).mean().item(),
            "SSIM": ssim(y, z, window_size=5).mean().item(),
            "Loss": criterion(y, z).item(),
        }


def train(
    model,
    criterion,
    epochs,
    batch_size,
    save_interval,
    device,
    training_dataset,
    validation_dataset,
    checkpoint_folder,
    checkpoint_name,
    learning_rate,
    resume,
):
    log_dir = checkpoint_folder / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(model, SRCNN):
        params = [
            {"params": model.conv_1.parameters()},
            {"params": model.conv_2.parameters()},
            {"params": model.conv_3.parameters(), "lr": learning_rate * 0.1},
        ]
    else:
        params = model.parameters()

    optimizer = Adam(params, lr=learning_rate)

    if resume:
        checkpoint = torch.load(resume)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start = checkpoint["epoch"]
        history = checkpoint["history"]
    else:
        start = 0
        history = []

    training_dataset = PairsDataset(training_dataset)
    training_loader = DataLoader(
        training_dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    validation_dataset = PairsDataset(validation_dataset)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    with trange(1 + start, 1 + epochs, unit="epoch") as progress:
        for epoch in progress:
            should_save = epoch % save_interval == 0
            name = checkpoint_name.format(model=model.__class__.__name__, epoch=epoch)

            model.train()

            for batch, (x, y) in enumerate(training_loader):
                progress.set_description(f"Training batch {batch}")

                optimizer.zero_grad()

                y = y.to(device)
                z = model(x.to(device))

                loss = criterion(z, y)

                loss.backward()
                optimizer.step()

                if batch == 0:
                    history.append(stats(epoch, "Training", criterion, y, z))

            with torch.no_grad():
                for batch, (x, y) in enumerate(validation_loader):
                    progress.set_description(f"Validating batch {batch}")

                    y = y.to(device)
                    z = model(x.to(device))

                    if should_save and batch == 0:
                        save_image(make_grid(y), (log_dir / name).with_suffix(".Y.png"))
                        save_image(make_grid(z), (log_dir / name).with_suffix(".Ŷ.png"))

                    history.append(stats(epoch, "Validation", criterion, y, z))

            if should_save:
                state = {
                    "epoch": epoch,
                    "history": history,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                torch.save(state, checkpoint_folder / name)

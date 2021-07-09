# %%
# !pip install --quiet h5py kornia pytorch-lightning scikit-learn torch torchvision wandb
# !wandb login

# %%

from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import torch
from kornia.losses import psnr as psnr_score
from kornia.losses import ssim as ssim_score
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Conv2d, Module, MSELoss, PixelShuffle, Sequential, Tanh
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize

# %%

try:
    from google.colab import drive
except ImportError:
    pass
else:
    drive.mount("/content/drive/", force_remount=True)


# %%
# !ln -s /content/drive/MyDrive/ var/

# %%
# !cp var/rphln-danbooru2020-small.{train,test}.h5 /dev/shm

# %%


class ESPCN(Sequential):
    SCALE = 4
    N0 = 3

    F1 = 5
    N1 = 64

    F2 = 3
    N2 = 32

    F3 = 3
    N3 = N0 * (SCALE ** 2)

    def __init__(self):
        super().__init__()

        self.stem = Sequential(
            Conv2d(self.N0, self.N1, self.F1, padding="same"),
            Tanh(),
            Conv2d(self.N1, self.N2, self.F2, padding="same"),
            Tanh(),
        )
        self.head = Sequential(
            Conv2d(self.N2, self.N3, self.F3, padding="same"),
            PixelShuffle(4),
        )


# %%


def mean_psnr(u: Tensor, v: Tensor) -> Tensor:
    return psnr_score(u, v, 1).mean()


def mean_ssim(u: Tensor, v: Tensor) -> Tensor:
    return ssim_score(u, v, 9).mean()


class LitModel(LightningModule):
    def __init__(self, model: Module, criterion: Module, learning_rate: float = 0.001):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.criterion = criterion

    def forward(self, x: Tensor) -> Tensor:
        x = normalize(x, std=[0.2931, 0.2985, 0.2946], mean=[0.7026, 0.6407, 0.6265])
        return self.model(x)

    def training_step(self, batch, index):
        x, y = batch
        z = self(x)

        loss = self.criterion(z, y)
        self.log("Training/Loss", loss)

        return loss

    def validation_step(self, batch, index):
        x, y = batch
        z = self(x)

        loss = self.criterion(z, y)
        self.log("Validation/Loss", loss)

        psnr = mean_psnr(z, y)
        self.log("Validation/PSNR", psnr)

        ssim = mean_ssim(z, y)
        self.log("Validation/SSIM", ssim)

    def test_step(self, batch, index):
        x, y = batch
        z = self(x)

        psnr = mean_psnr(z, y)
        self.log("Testing/PSNR", psnr)

        ssim = mean_ssim(z, y)
        self.log("Testing/SSIM", ssim)

    def configure_optimizers(self):
        parameters = [
            {"params": self.model.stem.parameters(), "lr": self.learning_rate},
            {"params": self.model.head.parameters(), "lr": self.learning_rate * 0.1},
        ]

        return Adam(parameters)


# %%


class TensorPairsDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, name):
        with h5py.File(name, "r") as h5:
            self.lr = torch.as_tensor(h5["lr"][:])
            self.hr = torch.as_tensor(h5["hr"][:])

    def __getitem__(self, index: int):
        lr = self.lr[index]
        hr = self.hr[index]

        return lr, hr

    def __len__(self):
        return len(self.lr)


class TensorPairsDataModule(LightningDataModule):
    batch_size: int

    train_with: Union[Path, PathLike, str]
    test_with: Union[Path, PathLike, str]

    training: TensorPairsDataset
    validation: TensorPairsDataset
    testing: TensorPairsDataset

    test_ratio: float = 0.2

    drop_last: bool = True
    pin_memory: bool = True

    def __init__(
        self,
        train_with: Union[Path, PathLike, str],
        test_with: Union[Path, PathLike, str],
        batch_size: int,
        test_ratio: float = test_ratio,
        drop_last: bool = drop_last,
        pin_memory: bool = pin_memory,
    ):
        super().__init__()

        self.train_with = train_with
        self.test_with = test_with

        self.batch_size = batch_size

        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.test_ratio = test_ratio

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.training, self.validation = train_test_split(
                TensorPairsDataset(self.train_with),
                test_size=self.test_ratio,
                shuffle=False,
            )

        if stage in (None, "test"):
            self.testing = TensorPairsDataset(self.test_with)

    def train_dataloader(self):
        return DataLoader(
            self.training,
            self.batch_size,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            self.batch_size,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testing,
            self.batch_size,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )


# %%

config = {
    "BATCH_SIZE": 1024,
    "LEARNING_RATE": 0.001,
}

model = LitModel(
    model=ESPCN(),
    criterion=MSELoss(),
    learning_rate=config["LEARNING_RATE"],
)
data = TensorPairsDataModule(
    batch_size=config["BATCH_SIZE"],
    test_with="/dev/shm/rphln-danbooru2020-small.test.h5",
    train_with="/dev/shm/rphln-danbooru2020-small.train.h5",
)

wandb = WandbLogger(project="Cardboard4", config=config, save_dir="var")
wandb.watch(model)

trainer = Trainer(
    gpus=1,
    precision=16,
    logger=wandb,
    callbacks=[
        ModelCheckpoint(monitor="Validation/Loss", dirpath=wandb.save_dir),
        EarlyStopping(monitor="Validation/Loss"),
    ],
)

trainer.fit(model, datamodule=data)
trainer.test(datamodule=data)

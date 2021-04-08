import h5py
import torch
from torch.utils.data import Dataset


class TensorPairsDataset(Dataset):
    def __init__(self, file):
        self.h5 = h5py.File(file, "r", libver="latest", swmr=True)

    def __getitem__(self, index):
        lr = torch.from_numpy(self.h5["lr"][index])
        hr = torch.from_numpy(self.h5["hr"][index])

        return lr, hr

    def __len__(self):
        return len(self.h5["lr"])

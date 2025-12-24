import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class H5WindowsDataset(Dataset):
    """Dataset for loading windowed EEG data from HDF5 file."""

    def __init__(self, h5_path, idxs=None):
        self.h5_path = h5_path
        self._h5 = None

        with h5py.File(h5_path, "r") as h5:
            data_shape = h5["data"].shape
            a, b = data_shape[1], data_shape[2]
            if b >= 256 and a <= 256:
                self._stored_order = "C_T"
            elif a >= 256 and b <= 256:
                self._stored_order = "T_C"
            else:
                self._stored_order = "T_C" if a > b else "C_T"

            self.length = data_shape[0]

        if idxs is None:
            self.idxs = np.arange(self.length)
        else:
            self.idxs = np.array(idxs, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, i):
        self._ensure_h5()
        idx = int(self.idxs[i])
        x = self._h5["data"][idx]
        y = int(self._h5["labels"][idx])

        if self._stored_order == "T_C":
            x = x.T.astype(np.float32)
        else:
            x = x.astype(np.float32)
        return x, y


def collate_h5(batch):
    """Collate for H5WindowsDataset"""
    xs, ys = zip(*batch)
    xs = np.stack(xs, axis=0)
    xs = torch.from_numpy(xs).float()
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys


def make_loader_h5(h5_path, idxs=None, batch_size=32, shuffle=True, num_workers=0):
    ds = H5WindowsDataset(h5_path, idxs=idxs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_h5,
        persistent_workers=(num_workers > 0),
    )
    return loader

import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingH5Dataset(Dataset):
    """Dataset for loading embeddings from HDF5 cache."""

    """Optimized for faster access."""

    def __init__(self, cache_h5, idxs=None):
        self.cache_h5 = cache_h5
        self._h5 = None
        with h5py.File(cache_h5, "r") as hf:
            total = hf["temb"].shape[0]
        self.length = total
        self.idxs = np.arange(total) if idxs is None else np.array(idxs, dtype=np.int64)

    def _ensure(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.cache_h5, "r")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        self._ensure()
        idx = int(self.idxs[i])
        temb = self._h5["temb"][idx]
        zfreq = self._h5["zfreq"][idx]
        label = int(self._h5["labels"][idx])
        return temb.astype(np.float32), zfreq.astype(np.float32), label


def collate_embeddings(batch):
    """Collate for EmbeddingH5Dataset."""
    tembs, zfreqs, labels = zip(*batch)
    tembs = torch.from_numpy(np.stack(tembs, axis=0))
    zfreqs = torch.from_numpy(np.stack(zfreqs, axis=0))
    labels = torch.tensor(labels, dtype=torch.long)
    return tembs, zfreqs, labels


def make_embedding_loader(
    cache_h5, idxs=None, batch_size=32, shuffle=True, num_workers=0
):
    ds = EmbeddingH5Dataset(cache_h5, idxs=idxs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_embeddings,
        persistent_workers=(num_workers > 0),
    )
    return loader

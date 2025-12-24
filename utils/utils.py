import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.h5_dataset import H5WindowsDataset, collate_h5


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_femba_embeddings(model, batch_x, device, emb_pool="mean"):
    """Extract temporal embeddings from FEMBA"""
    model.eval()
    with torch.no_grad():
        xb = batch_x.to(device)
        x = model.patch_embed(xb)
        pe = model.pos_embed.to(x.device, x.dtype)

        # Interpolate positional embedding if needed
        if x.shape[1] != pe.shape[1]:
            pe = F.interpolate(
                pe.permute(0, 2, 1), size=x.shape[1], mode="linear", align_corners=False
            ).permute(0, 2, 1)

        x = x + pe

        for mamba_block, norm_layer in zip(model.mamba_blocks, model.norm_layers):
            res = x
            x = mamba_block(x)
            x = res + x
            x = norm_layer(x)
        if emb_pool == "mean":
            emb = x.mean(dim=1)
        elif emb_pool == "max":
            emb, _ = x.max(dim=1)
    return emb


def infer_femba_emb_dim(femba_model, h5_path, device):
    """Infer FEMBA embedding dimension"""
    femba_model.eval()
    ds = H5WindowsDataset(h5_path, idxs=[0])
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_h5)
    xb, _ = next(iter(loader))
    emb = compute_femba_embeddings(femba_model, xb, device=device)
    dim = emb.shape[1]
    del xb, emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(dim)


def make_data_splits(h5_path, val_ratio=0.2, seed=42):
    """Create train/val split"""
    with h5py.File(h5_path, "r") as hf:
        N = hf["data"].shape[0]

    rng = np.random.default_rng(seed)
    all_idxs = np.arange(N)
    rng.shuffle(all_idxs)

    split = int((1 - val_ratio) * N)
    train_idxs = all_idxs[:split]
    val_idxs = all_idxs[split:]

    return train_idxs, val_idxs


def compute_spectrogram(x, cfg, device, eps=1e-12):
    """Compute spectrogram using STFT"""
    n_fft = cfg.model.stft.n_fft
    win_length = cfg.model.stft.n_per_seg
    hop_length = cfg.model.stft.n_per_seg - cfg.model.stft.n_overlap
    window = torch.hann_window(win_length, device=device)

    B, C, T = x.shape
    x_bc = x.reshape(B * C, T)

    S = torch.stft(
        x_bc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )

    P = S.abs() ** 2
    P_mean_time = P.mean(dim=-1)
    P_mean_channels = P_mean_time.reshape(B, C, -1).mean(dim=1)
    p_log = torch.log1p(P_mean_channels + eps)
    return p_log

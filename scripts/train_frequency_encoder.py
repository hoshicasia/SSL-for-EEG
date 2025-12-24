import os
import random
import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.h5_dataset import H5WindowsDataset, collate_h5
from models.femba import load_pretrained_femba
from models.frequency_encoder import FrequencyEncoderMAE
from utils.utils import (
    compute_femba_embeddings,
    compute_spectrogram,
    infer_femba_emb_dim,
    make_data_splits,
    set_seed,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_frequency_encoder(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"current device: {device}")
    set_seed(cfg.seed)

    femba = load_pretrained_femba(cfg.data.femba_model_path, device)

    n_freqs = cfg.model.stft.n_fft // 2 + 1
    femba_emb_dim = infer_femba_emb_dim(femba, cfg.data.h5_path, device)
    freq_latent = max(64, femba_emb_dim // 2)

    freq_encoder = FrequencyEncoderMAE(
        n_freqs=n_freqs,
        latent_dim=freq_latent,
        hidden=cfg.model.architecture.freq_hidden,
        proj_dim=cfg.model.architecture.proj_dim,
    ).to(device)

    time_proj = nn.Sequential(
        nn.Linear(femba_emb_dim, cfg.model.architecture.proj_dim),
        nn.ReLU(),
        nn.Linear(cfg.model.architecture.proj_dim, cfg.model.architecture.proj_dim),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(freq_encoder.parameters()) + list(time_proj.parameters()),
        lr=cfg.training.learning_rate,
        betas=cfg.training.optimizer.betas,
        eps=cfg.training.optimizer.eps,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    train_idxs, val_idxs = make_data_splits(
        cfg.data.h5_path, val_ratio=cfg.data.val_ratio, seed=cfg.seed
    )

    train_ds = H5WindowsDataset(cfg.data.h5_path, idxs=train_idxs)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_h5,
    )

    scaler = GradScaler(enabled=cfg.training.amp.enabled and device.type == "cuda")
    for epoch in range(1, cfg.training.epochs + 1):
        freq_encoder.train()
        time_proj.train()

        running_loss = 0.0
        running_recon_loss = 0.0
        running_consistency_loss = 0.0
        n_samples = 0

        for xb, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"):
            xb = xb.to(device, non_blocking=True).float()
            B = xb.shape[0]

            with torch.no_grad():
                spec = compute_spectrogram(xb, cfg, device)
                mean = spec.mean(dim=1, keepdim=True)
                std = spec.std(dim=1, keepdim=True).clamp(min=1e-6)
                spec_norm = (spec - mean) / std

            mask = (
                torch.rand(B, n_freqs, device=device) < cfg.training.masking.mask_prob
            )
            x_masked = spec_norm.clone()
            x_masked[mask] = 0.0
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.training.amp.enabled and device.type == "cuda"):
                recon, z_freq, proj_freq = freq_encoder(x_masked)
                emb_time = compute_femba_embeddings(femba, xb, device)
                proj_time = time_proj(emb_time)
                mask_float = mask.to(dtype=recon.dtype)
                se = (recon - spec_norm) ** 2
                masked_sum = (se * mask_float).sum()
                mask_count = mask_float.sum()
                recon_loss = masked_sum / (mask_count + 1e-12)
                cos = F.cosine_similarity(proj_freq, proj_time, dim=1)
                consistency_loss = (1.0 - cos).mean()
                loss = (
                    cfg.training.loss.recon_weight * recon_loss
                    + cfg.training.loss.consistency_weight * consistency_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * B
            running_recon_loss += recon_loss.item() * B
            running_consistency_loss += consistency_loss.item() * B
            n_samples += B

        avg_loss = running_loss / max(1, n_samples)
        avg_recon_loss = running_recon_loss / max(1, n_samples)
        avg_consistency_loss = running_consistency_loss / max(1, n_samples)

        print(f"  loss: {avg_loss}")
        print(f"  reconstruction loss : {avg_recon_loss}")
        print(f"  consistency Loss: {avg_consistency_loss}")

        if epoch % cfg.training.checkpoint.save_every == 0:
            ckpt_path = Path(cfg.data.output_dir) / f"ssl_checkpoint_epoch_{epoch}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "freq_encoder": freq_encoder.state_dict(),
                    "time_proj": time_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "metrics": {
                        "loss": avg_loss,
                        "recon_loss": avg_recon_loss,
                        "consistency_loss": avg_consistency_loss,
                    },
                },
                ckpt_path,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return freq_encoder, time_proj


if __name__ == "__main__":
    train_frequency_encoder()

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
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


def compute_freq_embeddings(freq_encoder, xb, cfg, device):
    """Extract frequency embeddings from spectrogram"""
    with torch.no_grad():
        spec = compute_spectrogram(xb, cfg, device)
        mean = spec.mean(dim=1, keepdim=True)
        std = spec.std(dim=1, keepdim=True).clamp(min=1e-6)
        spec_norm = (spec - mean) / std
        _, z_freq, _ = freq_encoder(spec_norm)
    return z_freq


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def finetune_concatenation(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    femba = load_pretrained_femba(cfg.data.femba_model_path, device)
    femba.eval()
    for param in femba.parameters():
        param.requires_grad = False

    n_freqs = cfg.model.stft.n_fft // 2 + 1
    femba_emb_dim = infer_femba_emb_dim(femba, cfg.data.h5_path, device)
    freq_latent = max(64, femba_emb_dim // 2)

    freq_encoder = FrequencyEncoderMAE(
        n_freqs=n_freqs,
        latent_dim=freq_latent,
        hidden=cfg.model.architecture.freq_hidden,
        proj_dim=cfg.model.architecture.proj_dim,
    ).to(device)

    checkpoint_path = cfg.model.freq_encoder_checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    freq_encoder.load_state_dict(ckpt["freq_encoder"])
    freq_encoder.eval()
    for param in freq_encoder.parameters():
        param.requires_grad = False

    print(f"freq_encoder from {checkpoint_path}")

    combined_dim = femba_emb_dim + freq_latent
    classifier = nn.Sequential(
        nn.Linear(combined_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    ).to(device)

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

    val_ds = H5WindowsDataset(cfg.data.h5_path, idxs=val_idxs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_h5,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_balanced_acc = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        classifier.train()
        running_loss = 0.0
        n_samples = 0

        for xb, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}"
        ):
            xb = xb.to(device, non_blocking=True).float()
            labels = labels.to(device)
            B = xb.shape[0]

            with torch.no_grad():
                emb_time = compute_femba_embeddings(femba, xb, device)
                emb_freq = compute_freq_embeddings(freq_encoder, xb, cfg, device)

            emb_combined = torch.cat([emb_time, emb_freq], dim=1)

            optimizer.zero_grad()
            logits = classifier(emb_combined)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            n_samples += B

        classifier.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for xb, labels in tqdm(val_loader, desc="Validating"):
                xb = xb.to(device, non_blocking=True).float()
                labels = labels.to(device)
                B = xb.shape[0]

                emb_time = compute_femba_embeddings(femba, xb, device)
                emb_freq = compute_freq_embeddings(freq_encoder, xb, cfg, device)
                emb_combined = torch.cat([emb_time, emb_freq], dim=1)

                logits = classifier(emb_combined)
                loss = criterion(logits, labels)

                val_loss += loss.item() * B
                val_samples += B

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        sensitivity = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {running_loss/n_samples:.4f}")
        print(f"  Val loss: {val_loss/val_samples:.4f}")
        print(f"  Val accuracy: {acc:.4f}")
        print(f"  Val balanced accuracy: {balanced_acc:.4f}")
        print(f"  Val F1 (weighted): {f1_weighted:.4f}")
        print(f"  Val F1 (macro): {f1_macro:.4f}")
        print(f"  Val sensitivity: {sensitivity:.4f}")

        scheduler.step(balanced_acc)

    print(f"\nBest balanced accuracy: {best_balanced_acc:.4f}")
    return classifier


if __name__ == "__main__":
    finetune_concatenation()

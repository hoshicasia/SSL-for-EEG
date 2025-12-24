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
from utils.utils import (
    compute_femba_embeddings,
    infer_femba_emb_dim,
    make_data_splits,
    set_seed,
)


def compute_class_weights(h5_path, idxs):
    with h5py.File(h5_path, "r") as hf:
        labels = hf["labels"][idxs]
    unique, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(unique) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def validate(classifier, femba, val_loader, criterion, device):
    """Validation loop with metrics"""
    classifier.eval()

    all_preds = []
    all_labels = []
    val_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for xb, labels in tqdm(val_loader, desc="Validating"):
            xb = xb.to(device, non_blocking=True).float()
            labels = labels.to(device)
            B = xb.shape[0]

            emb = compute_femba_embeddings(femba, xb, device)

            logits = classifier(emb)
            loss = criterion(logits, labels)

            val_loss += loss.item() * B
            n_samples += B

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if len(np.unique(all_labels)) > 1:
        sensitivity = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    else:
        sensitivity = 0.0

    avg_loss = val_loss / max(1, n_samples)

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "sensitivity": sensitivity,
    }

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_baseline_frozen_femba(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    femba = load_pretrained_femba(cfg.data.femba_model_path, device)
    femba.eval()
    for param in femba.parameters():
        param.requires_grad = False

    femba_emb_dim = infer_femba_emb_dim(femba, cfg.data.h5_path, device)
    classifier = nn.Sequential(
        nn.Linear(femba_emb_dim, 256),
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

    class_weights = compute_class_weights(cfg.data.h5_path, train_idxs).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    print("Start train")

    best_balanced_acc = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.training.epochs + 1):
        classifier.train()

        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
        for xb, labels in pbar:
            xb = xb.to(device, non_blocking=True).float()
            labels = labels.to(device)
            B = xb.shape[0]

            with torch.no_grad():
                emb = compute_femba_embeddings(femba, xb, device)
            optimizer.zero_grad()
            logits = classifier(emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            n_samples += B

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, n_samples)
        val_metrics = validate(classifier, femba, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        print(f"  Train loss:          {avg_train_loss:.4f}")
        print(f"  Val loss:            {val_metrics['loss']:.4f}")
        print(f"  Val accuracy:        {val_metrics['accuracy']:.4f}")
        print(f"  Val balanced accuracy:    {val_metrics['balanced_accuracy']:.4f}")
        print(f"  Val F1 (weighted):   {val_metrics['f1_weighted']:.4f}")
        print(f"  Val F1 (macro):      {val_metrics['f1_macro']:.4f}")
        print(f"  Val sensitivity:     {val_metrics['sensitivity']:.4f}")

        if val_metrics["balanced_accuracy"] > best_balanced_acc:
            best_balanced_acc = val_metrics["balanced_accuracy"]
            best_epoch = epoch

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Best balanced accuracy: {best_balanced_acc:.4f} (Epoch {best_epoch})")
    return classifier


if __name__ == "__main__":
    train_baseline_frozen_femba()

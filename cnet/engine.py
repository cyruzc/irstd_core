"""
Training and evaluation engine for binary segmentation.

Provides:
- train_one_epoch:  single epoch training loop (with grad clip)
- evaluate_epoch:   fast IoU/nIoU eval (GPU-only, for per-epoch)
- evaluate_final:   full metrics IoU/nIoU/PD/FA (for final report)
- save_checkpoint / load_checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics import FastIoU, SegMetrics


# ============ Checkpoint ============

def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
    }
    if metrics is not None:
        state["metrics"] = metrics
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    map_location: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


# ============ Train ============

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============ Evaluate ============

@torch.no_grad()
def predict_prob(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass, returns (prob, mask) on device."""
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    logits = model(image)
    prob = torch.sigmoid(logits)
    # resize back if padded
    if prob.shape[-2:] != mask.shape[-2:]:
        prob = F.interpolate(prob, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return prob, mask


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Fast GPU-only IoU/nIoU for per-epoch validation."""
    model.eval()
    meter = FastIoU(threshold=threshold)
    for batch in loader:
        prob, mask = predict_prob(model, batch, device)
        meter.update(prob, mask)
    return meter.get()


@torch.no_grad()
def evaluate_final(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    distance_thresh: float = 3.0,
) -> dict[str, float]:
    """Full metrics: IoU, nIoU, PD, FA."""
    model.eval()
    meter = SegMetrics(threshold=threshold, distance_thresh=distance_thresh)
    for batch in loader:
        prob, mask = predict_prob(model, batch, device)
        meter.update(prob, mask)
    return meter.get()


# ============ Helpers ============

def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "train_one_epoch",
    "evaluate_epoch",
    "evaluate_final",
    "predict_prob",
    "save_checkpoint",
    "load_checkpoint",
    "save_json",
]

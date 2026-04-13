from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ellipse_losses import EllipseReconstructionLoss
from ellipse_renderer import raw_to_soft_mask
from ellipse_utils import angle_abs_error, CropMeta, paste_patch
from engine import save_checkpoint, save_json
from metrics import FastIoU, SegMetrics


@torch.no_grad()
def batch_param_errors(decoded: dict[str, torch.Tensor], gt_params: torch.Tensor) -> dict[str, float]:
    pred_dx = decoded["dx"].detach().cpu().numpy()
    pred_dy = decoded["dy"].detach().cpu().numpy()
    pred_a = decoded["a"].detach().cpu().numpy()
    pred_b = decoded["b"].detach().cpu().numpy()
    pred_phi = decoded["phi"].detach().cpu().numpy()
    gt = gt_params.detach().cpu().numpy()
    angle_err = [angle_abs_error(float(p), float(g)) for p, g in zip(pred_phi, gt[:, 4])]
    return {
        "mae_dx": float(np.mean(np.abs(pred_dx - gt[:, 0]))),
        "mae_dy": float(np.mean(np.abs(pred_dy - gt[:, 1]))),
        "mae_a": float(np.mean(np.abs(pred_a - gt[:, 2]))),
        "mae_b": float(np.mean(np.abs(pred_b - gt[:, 3]))),
        "mae_phi": float(np.mean(angle_err)),
    }


def forward_batch(model: torch.nn.Module, batch: dict, patch_size: int, device: torch.device, use_amp: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    image = batch["image"].to(device, non_blocking=True)
    center_hint = batch["center_hint"].to(device, non_blocking=True)
    with autocast(device_type=device.type, enabled=use_amp):
        raw = model(image, center_hint)
    pred_mask, decoded = raw_to_soft_mask(raw, patch_size=patch_size)
    return pred_mask, decoded, raw


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: EllipseReconstructionLoss, device: torch.device, scaler: GradScaler, patch_size: int, use_amp: bool) -> dict[str, float]:
    model.train()
    stats_sum: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        gt_mask = batch["mask"].to(device, non_blocking=True)
        gt_params = batch["gt_params"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            raw = model(batch["image"].to(device, non_blocking=True), batch["center_hint"].to(device, non_blocking=True))
            pred_mask, decoded = raw_to_soft_mask(raw, patch_size=patch_size)
            loss, loss_stats = criterion(pred_mask, gt_mask, decoded, gt_params)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for key, value in loss_stats.items():
            stats_sum[key] += float(value)
        count += 1

    return {key: value / max(count, 1) for key, value in stats_sum.items()}


@torch.no_grad()
def evaluate_instance_level(model: torch.nn.Module, loader: DataLoader, device: torch.device, patch_size: int, use_amp: bool, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    meter = FastIoU(threshold=threshold)
    aggregated_errors: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        gt_mask = batch["mask"].to(device, non_blocking=True)
        gt_params = batch["gt_params"].to(device, non_blocking=True)
        raw = model(batch["image"].to(device, non_blocking=True), batch["center_hint"].to(device, non_blocking=True))
        pred_mask, decoded = raw_to_soft_mask(raw, patch_size=patch_size)
        meter.update(pred_mask, gt_mask)
        errors = batch_param_errors(decoded, gt_params)
        for key, value in errors.items():
            aggregated_errors[key] += value
        count += 1

    out = meter.get()
    out.update({key: value / max(count, 1) for key, value in aggregated_errors.items()})
    return out


@torch.no_grad()
def evaluate_full_images(model: torch.nn.Module, loader: DataLoader, device: torch.device, patch_size: int, use_amp: bool, threshold: float = 0.5, distance_thresh: float = 3.0) -> dict[str, float]:
    model.eval()
    pred_canvases: dict[str, np.ndarray] = {}
    gt_canvases: dict[str, np.ndarray] = {}

    for batch in loader:
        raw = model(batch["image"].to(device, non_blocking=True), batch["center_hint"].to(device, non_blocking=True))
        pred_mask, _ = raw_to_soft_mask(raw, patch_size=patch_size)
        pred_np = pred_mask.detach().cpu().numpy()[:, 0]
        gt_np = batch["mask"].detach().cpu().numpy()[:, 0]
        names = batch["name"]

        for i, name in enumerate(names):
            image_h = int(batch["image_h"][i].item())
            image_w = int(batch["image_w"][i].item())
            if name not in pred_canvases:
                pred_canvases[name] = np.zeros((image_h, image_w), dtype=np.float32)
                gt_canvases[name] = np.zeros((image_h, image_w), dtype=np.float32)

            meta = CropMeta(
                src_top=int(batch["src_top"][i].item()),
                src_left=int(batch["src_left"][i].item()),
                src_bottom=int(batch["src_bottom"][i].item()),
                src_right=int(batch["src_right"][i].item()),
                dst_top=int(batch["dst_top"][i].item()),
                dst_left=int(batch["dst_left"][i].item()),
                patch_size=int(batch["patch_size"][i].item()),
                image_h=image_h,
                image_w=image_w,
            )
            paste_patch(pred_canvases[name], pred_np[i], meta, reduce="max")
            paste_patch(gt_canvases[name], gt_np[i], meta, reduce="max")

    metric = SegMetrics(threshold=threshold, distance_thresh=distance_thresh)
    for name in pred_canvases:
        pred_tensor = torch.from_numpy(pred_canvases[name]).view(1, 1, *pred_canvases[name].shape)
        gt_tensor = torch.from_numpy(gt_canvases[name]).view(1, 1, *gt_canvases[name].shape)
        metric.update(pred_tensor, gt_tensor)
    return metric.get()


def maybe_save_best(checkpoint_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float], best_score: float, score_key: str = "IoU") -> float:
    score = float(metrics.get(score_key, 0.0))
    save_checkpoint(checkpoint_dir / "last.pt", model=model, epoch=epoch, metrics=metrics, optimizer=optimizer)
    save_json(checkpoint_dir / "last_metrics.json", metrics)
    if score >= best_score:
        save_checkpoint(checkpoint_dir / "best.pt", model=model, epoch=epoch, metrics=metrics, optimizer=optimizer)
        save_json(checkpoint_dir / "best_metrics.json", metrics)
        return score
    return best_score


__all__ = [
    "train_one_epoch",
    "evaluate_instance_level",
    "evaluate_full_images",
    "maybe_save_best",
]

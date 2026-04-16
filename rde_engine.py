from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ellipse_utils import angle_abs_error, CropMeta, paste_patch
from engine import save_checkpoint, save_json
from metrics import FastIoU, SegMetrics
from rde_losses import RadialProfileEllipseLoss
from rde_renderer import compute_gt_radial_profile, raw_to_soft_rde


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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: RadialProfileEllipseLoss,
    device: torch.device,
    patch_size: int,
    num_profile_samples: int,
    temperature: float = 12.0,
    current_epoch: int = 0,
    pretrain_base: bool = False,
) -> dict[str, float]:
    model.train()
    stats_sum: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        gt_mask = batch["mask"].to(device, non_blocking=True)
        gt_params = batch["gt_params"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        raw = model(batch["image"].to(device, non_blocking=True), batch["center_hint"].to(device, non_blocking=True))
        pred_mask, decoded = raw_to_soft_rde(raw, patch_size=patch_size, num_profile_samples=num_profile_samples, temperature=temperature)

        # Compute GT radial profile
        gt_profile = compute_gt_radial_profile(gt_mask, decoded, patch_size, num_profile_samples)

        loss, loss_stats = criterion(
            pred_mask, gt_mask, decoded, gt_params,
            gt_profile=gt_profile, current_epoch=current_epoch,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key, value in loss_stats.items():
            stats_sum[key] += float(value)
        count += 1

    return {key: value / max(count, 1) for key, value in stats_sum.items()}


@torch.no_grad()
def evaluate_instance_level(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    num_profile_samples: int,
    temperature: float = 12.0,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    meter = FastIoU(threshold=threshold)
    aggregated_errors: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        gt_mask = batch["mask"].to(device, non_blocking=True)
        gt_params = batch["gt_params"].to(device, non_blocking=True)
        raw = model(batch["image"].to(device, non_blocking=True), batch["center_hint"].to(device, non_blocking=True))
        pred_mask, decoded = raw_to_soft_rde(raw, patch_size=patch_size, num_profile_samples=num_profile_samples, temperature=temperature)
        meter.update(pred_mask, gt_mask)
        errors = batch_param_errors(decoded, gt_params)
        for key, value in errors.items():
            aggregated_errors[key] += value
        count += 1

    out = meter.get()
    out.update({key: value / max(count, 1) for key, value in aggregated_errors.items()})
    return out


@torch.no_grad()
def evaluate_full_images(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    num_profile_samples: int,
    temperature: float = 12.0,
    threshold: float = 0.5,
    distance_thresh: float = 3.0,
) -> dict[str, float]:
    model.eval()
    pred_canvases: dict[str, np.ndarray] = {}
    gt_canvases: dict[str, np.ndarray] = {}

    for batch in loader:
        raw = model(batch["image"].to(device, non_blocking=True), batch["center_hint"].to(device, non_blocking=True))
        pred_mask, _ = raw_to_soft_rde(raw, patch_size=patch_size, num_profile_samples=num_profile_samples, temperature=temperature)
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

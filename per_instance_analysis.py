from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cde_renderer import make_cde_raw_init_from_mask, raw_to_soft_cde
from data import load_grayscale, resolve_full_records
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_utils import component_instances, crop_with_pad, fit_ellipse_from_mask


def _manual_bce(pred, target):
    eps = 1e-6
    p = pred.clamp(eps, 1.0 - eps)
    return -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p)).mean()


def _soft_dice_loss(pred, target):
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def _iou_from_masks(pred, gt):
    """Binary IoU between two numpy masks."""
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    inter = (p * g).sum()
    union = ((p + g) > 0).astype(np.float32).sum()
    return float(inter / max(union, 1e-6))


def oracle_ellipse_iou(mask_patch, patch_size):
    """Oracle ellipse fit IoU for a single instance patch."""
    if mask_patch.sum() < 1:
        return 0.0
    ellipse = fit_ellipse_from_mask(mask_patch)
    # Render ellipse
    from ellipse_utils import rasterize_ellipse_numpy
    pred = rasterize_ellipse_numpy(patch_size, patch_size, ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.phi)
    return _iou_from_masks(pred, mask_patch)


def oracle_cde_iou(mask_patch, patch_size, num_fourier_terms, start_k, deform_scale, temperature, device, steps=200, lr=0.05):
    """Oracle CDE fit IoU for a single instance patch."""
    if mask_patch.sum() < 1:
        return 0.0
    from cde_losses import fourier_regularizer
    target = torch.from_numpy(mask_patch.astype(np.float32)).view(1, 1, patch_size, patch_size).to(device)
    raw = make_cde_raw_init_from_mask(mask_patch, patch_size, num_fourier_terms, start_k, device=device)
    raw = raw.clone().detach().requires_grad_(True)
    from torch.optim import Adam
    optimizer = Adam([raw], lr=lr)
    best_iou = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred_mask, decoded = raw_to_soft_cde(raw, patch_size, num_fourier_terms, start_k, deform_scale=deform_scale, temperature=temperature)
        loss = _soft_dice_loss(pred_mask, target) + 0.25 * _manual_bce(pred_mask, target) + 1e-4 * fourier_regularizer(decoded, start_k)
        loss.backward()
        optimizer.step()
        pred_np = (pred_mask.detach().cpu().numpy()[0, 0] > 0.5).astype(np.float32)
        iou = _iou_from_masks(pred_np, mask_patch)
        if iou > best_iou:
            best_iou = iou
    return best_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Per-instance oracle analysis")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--num-fourier-terms", type=int, default=3)
    parser.add_argument("--start-k", type=int, default=3)
    parser.add_argument("--deform-scale", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)
    records = resolve_full_records(config.root, config.test_split, image_dir_name=config.test_image_dir, mask_dir_name=config.test_mask_dir)

    all_instances = []
    for record in records:
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        for inst in component_instances(mask):
            comp_mask = inst["mask"]
            cx, cy = float(inst["centroid_x"]), float(inst["centroid_y"])
            mask_patch, _ = crop_with_pad(comp_mask, cx, cy, args.patch_size, pad_value=0.0)
            if mask_patch.sum() < 1:
                continue

            # Ellipse oracle
            ellipse_iou = oracle_ellipse_iou(mask_patch, args.patch_size)
            # CDE oracle
            cde_iou = oracle_cde_iou(mask_patch, args.patch_size, args.num_fourier_terms, args.start_k, args.deform_scale, args.temperature, device)

            delta = cde_iou - ellipse_iou
            all_instances.append({
                "name": record.name,
                "instance_id": int(inst["instance_id"]),
                "area": int(inst["area"]),
                "ellipse_ub_iou": round(ellipse_iou, 6),
                "cde_ub_iou": round(cde_iou, 6),
                "delta_oracle": round(delta, 6),
            })

    # Sort by delta
    all_instances.sort(key=lambda x: x["delta_oracle"])
    n = len(all_instances)
    third = n // 3

    groups = {
        "low_gain": all_instances[:third],
        "mid_gain": all_instances[third:2 * third],
        "high_gain": all_instances[2 * third:],
    }

    summary = {}
    for group_name, instances in groups.items():
        ellipse_ious = [x["ellipse_ub_iou"] for x in instances]
        cde_ious = [x["cde_ub_iou"] for x in instances]
        deltas = [x["delta_oracle"] for x in instances]
        areas = [x["area"] for x in instances]
        summary[group_name] = {
            "count": len(instances),
            "mean_area": round(float(np.mean(areas)), 1),
            "mean_ellipse_ub": round(float(np.mean(ellipse_ious)), 4),
            "mean_cde_ub": round(float(np.mean(cde_ious)), 4),
            "mean_delta": round(float(np.mean(deltas)), 4),
        }

    output = {
        "num_instances": n,
        "overall_mean_ellipse_ub": round(float(np.mean([x["ellipse_ub_iou"] for x in all_instances])), 4),
        "overall_mean_cde_ub": round(float(np.mean([x["cde_ub_iou"] for x in all_instances])), 4),
        "overall_mean_delta": round(float(np.mean([x["delta_oracle"] for x in all_instances])), 4),
        "groups": summary,
        "instances": all_instances,
    }

    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(json.dumps({"num_instances": n, "groups": {k: {kk: vv for kk, vv in v.items() if kk != "instances"} for k, v in output.items() if k in ("groups",)}}, indent=2))

    # Print summary table
    print("\n=== Per-Instance Oracle Analysis ===")
    print(f"{'Group':<12} {'N':>5} {'Area':>8} {'EllipseUB':>10} {'CDE-UB':>10} {'Delta':>8}")
    for g in ["low_gain", "mid_gain", "high_gain"]:
        s = summary[g]
        print(f"{g:<12} {s['count']:>5} {s['mean_area']:>8.1f} {s['mean_ellipse_ub']:>10.4f} {s['mean_cde_ub']:>10.4f} {s['mean_delta']:>+8.4f}")


if __name__ == "__main__":
    main()

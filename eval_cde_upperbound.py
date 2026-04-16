from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from cde_losses import fourier_regularizer, soft_dice_loss
from cde_renderer import make_cde_raw_init_from_mask, raw_to_soft_cde
from data import load_grayscale, resolve_full_records
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_utils import component_instances, crop_with_pad, paste_patch
from metrics import FastIoU, SegMetrics

EPS = 1e-6


def _manual_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Manual BCE that avoids CUDA assert in F.binary_cross_entropy."""
    p = pred.clamp(EPS, 1.0 - EPS)
    return -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p)).mean()


@torch.no_grad()
def _to_patch_tensor(mask_patch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_patch.astype(np.float32)).view(1, 1, *mask_patch.shape).to(device)


def fit_cde_patch_upperbound(
    mask_patch: np.ndarray,
    patch_size: int,
    num_fourier_terms: int,
    start_k: int,
    deform_scale: float,
    temperature: float,
    device: torch.device,
    steps: int = 300,
    lr: float = 5e-2,
    w_bce: float = 0.25,
    w_fourier: float = 1e-4,
) -> np.ndarray:
    target = _to_patch_tensor(mask_patch, device=device)

    raw = make_cde_raw_init_from_mask(
        mask_patch=mask_patch,
        patch_size=patch_size,
        num_fourier_terms=num_fourier_terms,
        start_k=start_k,
        device=device,
    )
    raw = raw.clone().detach().requires_grad_(True)

    optimizer = Adam([raw], lr=lr)
    best_loss = float("inf")
    best_pred = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        pred_mask, decoded = raw_to_soft_cde(
            raw=raw,
            patch_size=patch_size,
            num_fourier_terms=num_fourier_terms,
            start_k=start_k,
            deform_scale=deform_scale,
            temperature=temperature,
        )

        loss_dice = soft_dice_loss(pred_mask, target)
        loss_bce = _manual_bce(pred_mask, target)
        loss_fourier = fourier_regularizer(decoded, start_k=start_k)
        loss = loss_dice + w_bce * loss_bce + w_fourier * loss_fourier

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_pred = pred_mask.detach().clone()

    if best_pred is None:
        best_pred, _ = raw_to_soft_cde(
            raw=raw.detach(),
            patch_size=patch_size,
            num_fourier_terms=num_fourier_terms,
            start_k=start_k,
            deform_scale=deform_scale,
            temperature=temperature,
        )

    pred_np = best_pred.detach().cpu().numpy()[0, 0]
    return (pred_np > 0.5).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Deformable Ellipse upper bound for IR small targets")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--image-dir-name", type=str, default=None)
    parser.add_argument("--mask-dir-name", type=str, default=None)
    parser.add_argument("--test-image-dir-name", type=str, default=None)
    parser.add_argument("--test-mask-dir-name", type=str, default=None)

    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--num-fourier-terms", type=int, default=3)
    parser.add_argument("--start-k", type=int, default=3)
    parser.add_argument("--deform-scale", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=12.0)

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)

    records = resolve_full_records(
        config.root,
        config.test_split,
        image_dir_name=config.test_image_dir,
        mask_dir_name=config.test_mask_dir,
    )

    full_metric = SegMetrics(threshold=args.threshold, distance_thresh=args.distance_thresh)
    instance_metric = FastIoU(threshold=args.threshold)
    num_instances = 0

    for record in records:
        full_mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        pred_canvas = np.zeros_like(full_mask, dtype=np.float32)

        for inst in component_instances(full_mask):
            comp_mask = inst["mask"]
            cx = float(inst["centroid_x"])
            cy = float(inst["centroid_y"])

            mask_patch, crop_meta = crop_with_pad(
                comp_mask, center_x=cx, center_y=cy, patch_size=args.patch_size, pad_value=0.0,
            )

            pred_patch = fit_cde_patch_upperbound(
                mask_patch=mask_patch,
                patch_size=args.patch_size,
                num_fourier_terms=args.num_fourier_terms,
                start_k=args.start_k,
                deform_scale=args.deform_scale,
                temperature=args.temperature,
                device=device,
                steps=args.steps,
                lr=args.lr,
            )

            paste_patch(pred_canvas, pred_patch, crop_meta, reduce="max")

            instance_metric.update(
                torch.from_numpy(pred_patch).view(1, 1, *pred_patch.shape),
                torch.from_numpy(mask_patch).view(1, 1, *mask_patch.shape),
            )
            num_instances += 1

        full_metric.update(
            torch.from_numpy(pred_canvas).view(1, 1, *pred_canvas.shape),
            torch.from_numpy(full_mask).view(1, 1, *full_mask.shape),
        )

    metrics = {f"full_{k}": v for k, v in full_metric.get().items()}
    metrics.update({f"instance_{k}": v for k, v in instance_metric.get().items()})
    metrics["num_instances"] = num_instances
    metrics["note"] = (
        "Analytic upper bound for canonical deformable ellipse. "
        "GT instance masks are directly optimized by a base ellipse plus canonical Fourier residual."
    )
    metrics["num_fourier_terms"] = args.num_fourier_terms
    metrics["start_k"] = args.start_k
    metrics["deform_scale"] = args.deform_scale

    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

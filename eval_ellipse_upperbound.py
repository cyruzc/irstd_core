from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_grayscale, resolve_full_records
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_utils import component_instances, fit_ellipse_from_mask, rasterize_ellipse_numpy
from metrics import FastIoU, SegMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic ellipse-fit upper bound for centroid-conditioned ellipse reconstruction")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--image-dir-name", type=str, default=None)
    parser.add_argument("--mask-dir-name", type=str, default=None)
    parser.add_argument("--test-image-dir-name", type=str, default=None)
    parser.add_argument("--test-mask-dir-name", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        pred_canvas = np.zeros_like(mask, dtype=np.float32)
        gt_canvas = mask.astype(np.float32)

        for inst in component_instances(mask):
            comp_mask = inst["mask"]
            ellipse = fit_ellipse_from_mask(comp_mask)
            pred_comp = rasterize_ellipse_numpy(mask.shape[0], mask.shape[1], ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.phi)
            pred_canvas = np.maximum(pred_canvas, pred_comp)

            instance_metric.update(
                torch.from_numpy(pred_comp).view(1, 1, *pred_comp.shape),
                torch.from_numpy(comp_mask).view(1, 1, *comp_mask.shape),
            )
            num_instances += 1

        full_metric.update(
            torch.from_numpy(pred_canvas).view(1, 1, *pred_canvas.shape),
            torch.from_numpy(gt_canvas).view(1, 1, *gt_canvas.shape),
        )

    metrics = {f"full_{k}": v for k, v in full_metric.get().items()}
    metrics.update({f"instance_{k}": v for k, v in instance_metric.get().items()})
    metrics["num_instances"] = num_instances
    metrics["note"] = "Analytic ellipse fit to GT connected components; this is the ellipse-assumption upper bound, not an image-only predictor."

    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

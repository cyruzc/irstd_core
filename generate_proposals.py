"""Generate proposal dataset from point detector predictions.

Run point detector on train split, extract peaks, label each proposal
by distance to nearest GT centroid, and save to JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_grayscale, resolve_full_records
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_utils import component_instances, crop_with_pad, make_center_hint
from engine import load_checkpoint
from point_model import LiteUNet, extract_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="Generate proposal dataset for verifier training")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--point-checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, help="Split file name (default: train split from config)")
    parser.add_argument("--point-threshold", type=float, default=0.3)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--pos-radius", type=float, default=3.0, help="Max distance to GT for positive label")
    parser.add_argument("--neg-radius", type=float, default=8.0, help="Min distance to GT for negative label")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="runs/proposals.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=True)

    # Load point detector
    point_model = LiteUNet(in_channels=1).to(device)
    load_checkpoint(Path(args.point_checkpoint), point_model, map_location=str(device))
    point_model.eval()

    # Use train split by default, or override
    split_name = args.split or config.train_split
    if args.split:
        # If user specified a split, try to infer train/test dirs
        image_dir = config.train_image_dir
        mask_dir = config.train_mask_dir
    else:
        image_dir = config.train_image_dir
        mask_dir = config.train_mask_dir
    records = resolve_full_records(
        config.root, split_name,
        image_dir_name=image_dir, mask_dir_name=mask_dir,
    )

    all_proposals = []
    stats = {"total_images": 0, "total_proposals": 0, "positive": 0, "negative": 0, "ignored": 0}

    with torch.no_grad():
        for record in records:
            image = load_grayscale(record.image_path)
            image_norm = ((image - config.img_mean) / config.img_std).astype(np.float32)
            gt_mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)

            # Get GT centroids
            gt_centroids = [
                (float(inst["centroid_x"]), float(inst["centroid_y"]))
                for inst in component_instances(gt_mask)
            ]

            # Run detector
            image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_heatmap = point_model(image_tensor)
            peaks = extract_peaks(pred_heatmap[0], threshold=args.point_threshold, min_distance=args.min_distance)

            for px, py, score in peaks:
                # Compute distance to nearest GT centroid
                if gt_centroids:
                    min_dist = min((((px - gx) ** 2 + (py - gy) ** 2) ** 0.5) for gx, gy in gt_centroids)
                else:
                    min_dist = float("inf")

                # Label
                if min_dist <= args.pos_radius:
                    label = 1
                    stats["positive"] += 1
                elif min_dist >= args.neg_radius:
                    label = 0
                    stats["negative"] += 1
                else:
                    label = -1  # ignored
                    stats["ignored"] += 1

                all_proposals.append({
                    "image_path": str(record.image_path),
                    "mask_path": str(record.mask_path),
                    "name": record.name,
                    "x": float(px),
                    "y": float(py),
                    "score": float(score),
                    "distance_to_gt": float(min_dist),
                    "label": label,
                })

            stats["total_images"] += 1
            stats["total_proposals"] += len(peaks)

    # Save
    output = {
        "config": {
            "dataset_name": args.dataset_name,
            "split": split_name,
            "point_checkpoint": args.point_checkpoint,
            "point_threshold": args.point_threshold,
            "pos_radius": args.pos_radius,
            "neg_radius": args.neg_radius,
            "patch_size": args.patch_size,
        },
        "stats": stats,
        "proposals": all_proposals,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Generated {stats['total_proposals']} proposals from {stats['total_images']} images")
    print(f"  Positive: {stats['positive']}, Negative: {stats['negative']}, Ignored: {stats['ignored']}")


if __name__ == "__main__":
    main()

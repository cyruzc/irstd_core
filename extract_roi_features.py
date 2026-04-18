"""Pre-extract ROI features from LiteUNet backbone for all proposals.

Saves a single .pt file with all proposal features, scores, and labels.
Scorer training then becomes a fast MLP-only job.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data import load_grayscale
from dataset_config import build_dataset_config
from engine import load_checkpoint
from point_model import LiteUNet
from proposal_scorer import extract_roi_features


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-extract ROI features for proposals")
    parser.add_argument("--proposals", type=str, required=True)
    parser.add_argument("--point-checkpoint", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--roi-radius", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    img_mean, img_std = 0.0, 1.0
    if args.dataset_name:
        from dataset_config import DATASET_REGISTRY
        cfg = DATASET_REGISTRY[args.dataset_name]
        img_mean, img_std = cfg.img_mean, cfg.img_std

    # Load backbone
    backbone = LiteUNet(in_channels=1).to(device)
    load_checkpoint(Path(args.point_checkpoint), backbone, map_location=str(device))
    backbone.eval()
    print(f"Backbone loaded on {device}")

    # Load proposals
    data = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    proposals = [p for p in data["proposals"] if p["label"] >= 0]

    # Group by image
    by_image = defaultdict(list)
    for p in proposals:
        by_image[p["image_path"]].append(p)

    print(f"Processing {len(proposals)} proposals from {len(by_image)} images")

    all_feats = []
    all_scores = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for i, (img_path, props) in enumerate(by_image.items()):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(by_image)} images...")
            image = load_grayscale(img_path)
            image_norm = ((image - img_mean) / img_std).astype(np.float32)
            image_t = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)

            _, feat = backbone.forward_with_features(image_t)

            xs = torch.tensor([p["x"] for p in props], dtype=torch.float32, device=device)
            ys = torch.tensor([p["y"] for p in props], dtype=torch.float32, device=device)
            bidx = torch.zeros(len(props), dtype=torch.long, device=device)

            roi_feat = extract_roi_features(feat, bidx, xs, ys,
                                            roi_size=args.roi_size, radius=args.roi_radius)

            all_feats.append(roi_feat.cpu())
            all_scores.extend([p["score"] for p in props])
            all_labels.extend([p["label"] for p in props])
            all_names.extend([p["name"] for p in props])

    # Concatenate
    all_feats = torch.cat(all_feats, dim=0)
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.float32)

    output_path = args.output or args.proposals.replace(".json", "_roi.pt")
    torch.save({
        "feats": all_feats,
        "scores": all_scores,
        "labels": all_labels,
        "names": all_names,
        "config": {
            "roi_size": args.roi_size,
            "roi_radius": args.roi_radius,
            "dataset_name": args.dataset_name,
            "point_checkpoint": args.point_checkpoint,
        },
    }, output_path)

    pos = (all_labels == 1).sum().item()
    neg = (all_labels == 0).sum().item()
    print(f"Saved {output_path}: {len(all_labels)} proposals ({pos} pos, {neg} neg)")
    print(f"Feature shape: {all_feats.shape}")


if __name__ == "__main__":
    main()

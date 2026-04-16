from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset_config import build_dataset_config, validate_dataset_config
from engine import load_checkpoint, save_json
from point_data import build_point_datasets, PointDetectionDataset
from point_model import LiteUNet, extract_peaks


def make_loader(dataset: PointDetectionDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=_point_collate_fn)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        gt_heatmap = batch["heatmap"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_heatmap = model(image)

        # BCE loss
        loss = torch.nn.functional.binary_cross_entropy(pred_heatmap, gt_heatmap, reduction="none")
        # Weight positive pixels higher
        pos_weight = gt_heatmap.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
        neg_weight = (1 - gt_heatmap).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
        weight = gt_heatmap * (neg_weight / pos_weight) + (1 - gt_heatmap)
        loss = (loss * weight).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_point_detector(model, loader, device, threshold=0.3, min_distance=3, distance_thresh=3.0):
    model.eval()
    total_gt = 0
    total_detected = 0
    total_matched = 0
    total_distance_error = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        gt_heatmap = batch["heatmap"].to(device, non_blocking=True)
        num_points = batch["num_points"]

        pred_heatmap = model(image)

        for i in range(image.shape[0]):
            # GT points from heatmap
            gt_np = gt_heatmap[i, 0].cpu().numpy()
            gt_peaks = extract_peaks(torch.from_numpy(gt_np), threshold=0.3, min_distance=1)
            n_gt = int(num_points[i].item())
            total_gt += n_gt

            # Predicted points
            pred_peaks = extract_peaks(pred_heatmap[i], threshold=threshold, min_distance=min_distance)
            total_detected += len(pred_peaks)

            # Match predictions to GT (greedy nearest)
            gt_coords = [(p[0], p[1]) for p in gt_peaks[:n_gt]]
            pred_coords = [(p[0], p[1]) for p in pred_peaks]

            matched_pred = set()
            for gx, gy in gt_coords:
                best_dist = float("inf")
                best_j = -1
                for j, (px, py) in enumerate(pred_coords):
                    if j in matched_pred:
                        continue
                    d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                if best_j >= 0 and best_dist < distance_thresh:
                    total_matched += 1
                    matched_pred.add(best_j)
                    total_distance_error.append(best_dist)

    pd = total_matched / max(total_gt, 1)
    precision = total_matched / max(total_detected, 1)
    mean_error = float(np.mean(total_distance_error)) if total_distance_error else float("inf")
    return {"PD": pd, "precision": precision, "mean_distance_error": mean_error, "total_gt": total_gt, "total_detected": total_detected, "total_matched": total_matched}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LiteUNet point detector")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--image-dir-name", type=str, default=None)
    parser.add_argument("--mask-dir-name", type=str, default=None)
    parser.add_argument("--test-image-dir-name", type=str, default=None)
    parser.add_argument("--test-mask-dir-name", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="runs/point_detector")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def _point_collate_fn(batch):
    """Custom collate: pad images/heatmaps to same size within batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    images, heatmaps, names, num_points = [], [], [], []
    for b in batch:
        img = b["image"]
        hm = b["heatmap"]
        h, w = img.shape[1], img.shape[2]
        if h < max_h or w < max_w:
            pad_h, pad_w = max_h - h, max_w - w
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            hm = torch.nn.functional.pad(hm, (0, pad_w, 0, pad_h), value=0.0)
        images.append(img)
        heatmaps.append(hm)
        names.append(b["name"])
        num_points.append(b["num_points"])
    return {
        "image": torch.stack(images),
        "heatmap": torch.stack(heatmaps),
        "name": names,
        "num_points": torch.stack(num_points),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=not args.eval_only)

    train_set, val_set = build_point_datasets(config, sigma=args.sigma)
    val_loader = make_loader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LiteUNet(in_channels=1).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    if args.resume:
        load_checkpoint(Path(args.resume), model, map_location=str(device))
        print(f"Resumed from {args.resume}")

    if args.eval_only:
        metrics = evaluate_point_detector(model, val_loader, device, threshold=args.threshold)
        print(json.dumps(metrics, indent=2))
        return

    train_loader = make_loader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pd = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        metrics = evaluate_point_detector(model, val_loader, device, threshold=args.threshold)
        record = {"epoch": epoch, "loss": loss, "lr": float(optimizer.param_groups[0]["lr"]), **metrics}
        history.append(record)
        save_json(out_dir / "history.json", {"history": history})
        print(json.dumps(record, sort_keys=True))

        if metrics["PD"] >= best_pd:
            best_pd = metrics["PD"]
            from engine import save_checkpoint
            save_checkpoint(out_dir / "best.pt", model, epoch=epoch, metrics=metrics)

    save_json(out_dir / "best_metrics.json", max(history, key=lambda r: r.get("PD", 0)))


if __name__ == "__main__":
    main()

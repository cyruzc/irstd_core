"""Train ROI-based proposal scorer using shared LiteUNet backbone features.

Stage 1: Freeze backbone, train only scorer head.
Stage 2 (optional): Unfreeze for joint finetune.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import load_grayscale
from dataset_config import build_dataset_config, validate_dataset_config
from engine import load_checkpoint, save_checkpoint, save_json
from point_model import LiteUNet
from proposal_scorer import ROIScorer, extract_roi_features


class ProposalROIDataset:
    """Loads proposals grouped by image, returns image-level batches."""

    def __init__(
        self,
        proposals_path: str | Path,
        img_mean: float = 0.0,
        img_std: float = 1.0,
        balance: bool = True,
        seed: int = 42,
    ) -> None:
        self.img_mean = img_mean
        self.img_std = img_std
        self.rng = np.random.RandomState(seed)

        data = json.loads(Path(proposals_path).read_text(encoding="utf-8"))
        all_proposals = [p for p in data["proposals"] if p["label"] >= 0]

        # Group by image
        by_image = defaultdict(list)
        for p in all_proposals:
            by_image[p["image_path"]].append(p)

        self.image_groups = []
        self._image_cache: dict[str, np.ndarray] = {}

        pos_total = sum(1 for p in all_proposals if p["label"] == 1)
        neg_total = sum(1 for p in all_proposals if p["label"] == 0)

        for img_path, props in by_image.items():
            pos = [p for p in props if p["label"] == 1]
            neg = [p for p in props if p["label"] == 0]
            if balance:
                # Keep balanced within each image
                min_n = min(len(pos), len(neg))
                if min_n == 0 and (len(pos) > 0 or len(neg) > 0):
                    # Keep at least a few
                    if len(pos) > 0:
                        pos = pos[:min(3, len(pos))]
                        neg = neg[:min(len(pos) * 3, len(neg))]
                    else:
                        neg = neg[:min(5, len(neg))]
                else:
                    self.rng.shuffle(pos)
                    self.rng.shuffle(neg)
                    pos = pos[:min_n]
                    neg = neg[:min_n]
            self.image_groups.append((img_path, pos + neg))

        self.rng.shuffle(self.image_groups)

        kept_pos = sum(len([p for p in g[1] if p["label"] == 1]) for g in self.image_groups)
        kept_neg = sum(len([p for p in g[1] if p["label"] == 0]) for g in self.image_groups)
        print(f"ProposalROIDataset: {len(self.image_groups)} images, {kept_pos} pos + {kept_neg} neg")

    def _load_image(self, path: str) -> np.ndarray:
        if path not in self._image_cache:
            image = load_grayscale(path)
            self._image_cache[path] = ((image - self.img_mean) / self.img_std).astype(np.float32)
            if len(self._image_cache) > 200:
                oldest = next(iter(self._image_cache))
                del self._image_cache[oldest]
        return self._image_cache[path]


def train_one_epoch(backbone, scorer, dataset, optimizer, device, roi_size, roi_radius):
    backbone.eval()  # backbone always in eval mode (frozen)
    scorer.train()

    total_loss = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    total = 0

    indices = list(range(len(dataset.image_groups)))
    np.random.shuffle(indices)

    for idx in indices:
        img_path, proposals = dataset.image_groups[idx]
        if not proposals:
            continue

        image_norm = dataset._load_image(img_path)
        image_t = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            _, feat = backbone.forward_with_features(image_t)

        # Extract ROI features for all proposals
        xs = torch.tensor([p["x"] for p in proposals], dtype=torch.float32, device=device)
        ys = torch.tensor([p["y"] for p in proposals], dtype=torch.float32, device=device)
        scores = torch.tensor([p["score"] for p in proposals], dtype=torch.float32, device=device)
        labels = torch.tensor([p["label"] for p in proposals], dtype=torch.float32, device=device)
        batch_idx = torch.zeros(len(proposals), dtype=torch.long, device=device)

        roi_feat = extract_roi_features(feat, batch_idx, xs, ys, roi_size=roi_size, radius=roi_radius)

        optimizer.zero_grad(set_to_none=True)
        logits = scorer(roi_feat, scores)

        # Focal loss
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        prob = logits.detach().sigmoid()
        pt = labels * prob + (1 - labels) * (1 - prob)
        focal_weight = (1 - pt) ** 4.0
        loss = (2.0 * focal_weight * bce).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = (logits.detach().sigmoid() > 0.5).float()
        tp += ((pred == 1) & (labels == 1)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        tn += ((pred == 0) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()
        total += len(proposals)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return total_loss / max(len(indices), 1), f1, precision, recall


@torch.no_grad()
def evaluate(backbone, scorer, dataset, device, roi_size, roi_radius):
    backbone.eval()
    scorer.eval()

    tp, fp, tn, fn = 0, 0, 0, 0
    total_loss = 0.0
    count = 0

    for idx in range(len(dataset.image_groups)):
        img_path, proposals = dataset.image_groups[idx]
        if not proposals:
            continue

        image_norm = dataset._load_image(img_path)
        image_t = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        _, feat = backbone.forward_with_features(image_t)

        xs = torch.tensor([p["x"] for p in proposals], dtype=torch.float32, device=device)
        ys = torch.tensor([p["y"] for p in proposals], dtype=torch.float32, device=device)
        scores = torch.tensor([p["score"] for p in proposals], dtype=torch.float32, device=device)
        labels = torch.tensor([p["label"] for p in proposals], dtype=torch.float32, device=device)
        batch_idx = torch.zeros(len(proposals), dtype=torch.long, device=device)

        roi_feat = extract_roi_features(feat, batch_idx, xs, ys, roi_size=roi_size, radius=roi_radius)
        logits = scorer(roi_feat, scores)

        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        prob = logits.sigmoid()
        pt = labels * prob + (1 - labels) * (1 - prob)
        loss = (2.0 * (1 - pt) ** 4.0 * bce).mean()
        total_loss += loss.item()
        count += 1

        pred = (logits.sigmoid() > 0.5).float()
        tp += ((pred == 1) & (labels == 1)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        tn += ((pred == 0) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": total_loss / max(count, 1),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ROI proposal scorer")
    parser.add_argument("--proposals", type=str, required=True)
    parser.add_argument("--proposals-val", type=str, required=True)
    parser.add_argument("--point-checkpoint", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--roi-radius", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/scorer_roi")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    # Resolve normalization
    img_mean, img_std = 0.0, 1.0
    if args.dataset_name:
        from dataset_config import DATASET_REGISTRY
        cfg = DATASET_REGISTRY[args.dataset_name]
        img_mean, img_std = cfg.img_mean, cfg.img_std
        print(f"Using normalization: mean={img_mean}, std={img_std}")

    # Load backbone (frozen)
    backbone = LiteUNet(in_channels=1).to(device)
    load_checkpoint(Path(args.point_checkpoint), backbone, map_location=str(device))
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    print(f"Backbone loaded and frozen from {args.point_checkpoint}")

    # Scorer (trainable)
    scorer = ROIScorer(in_channels=16, hidden_dim=args.hidden_dim).to(device)
    trainable = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    print(f"Scorer params: {trainable}")

    train_set = ProposalROIDataset(args.proposals, img_mean=img_mean, img_std=img_std, balance=True, seed=args.seed)
    val_set = ProposalROIDataset(args.proposals_val, img_mean=img_mean, img_std=img_std, balance=False, seed=args.seed)

    optimizer = AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.batch_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss, f1, prec, rec = train_one_epoch(
            backbone, scorer, train_set, optimizer, device,
            roi_size=args.roi_size, roi_radius=args.roi_radius,
        )
        scheduler.step()

        val_metrics = evaluate(backbone, scorer, val_set, device,
                               roi_size=args.roi_size, roi_radius=args.roi_radius)

        record = {"epoch": epoch, "train_loss": loss, "train_f1": f1, "train_prec": prec, "train_rec": rec,
                  "lr": float(optimizer.param_groups[0]["lr"]), **val_metrics}
        history.append(record)

        print(f"Epoch {epoch}: loss={loss:.4f} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f} | "
              f"val_f1={val_metrics['f1']:.3f} val_prec={val_metrics['precision']:.3f} val_rec={val_metrics['recall']:.3f}")

        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(out_dir / "best.pt", scorer, epoch=epoch, metrics=val_metrics)

    save_json(out_dir / "best_metrics.json", max(history, key=lambda r: r.get("f1", 0)))
    save_json(out_dir / "history.json", {"history": history})


if __name__ == "__main__":
    main()

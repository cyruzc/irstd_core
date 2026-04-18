"""Train ROI scorer from pre-extracted features (MLP-only, very fast)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from proposal_scorer import ROIScorer
from engine import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train scorer from pre-extracted ROI features")
    parser.add_argument("--train-features", type=str, required=True)
    parser.add_argument("--val-features", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/scorer_roi")
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--no-balance", action="store_false", dest="balance")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted features
    train_data = torch.load(args.train_features, map_location="cpu", weights_only=False)
    val_data = torch.load(args.val_features, map_location="cpu", weights_only=False)

    train_feats, train_scores, train_labels = train_data["feats"], train_data["scores"], train_data["labels"]
    val_feats, val_scores, val_labels = val_data["feats"], val_data["scores"], val_data["labels"]

    print(f"Train: {len(train_labels)} ({(train_labels==1).sum()} pos, {(train_labels==0).sum()} neg)")
    print(f"Val:   {len(val_labels)} ({(val_labels==1).sum()} pos, {(val_labels==0).sum()} neg)")

    # Balance training set
    if args.balance:
        pos_idx = (train_labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (train_labels == 0).nonzero(as_tuple=True)[0]
        min_n = min(len(pos_idx), len(neg_idx))
        perm_pos = pos_idx[torch.randperm(len(pos_idx))[:min_n]]
        perm_neg = neg_idx[torch.randperm(len(neg_idx))[:min_n]]
        keep = torch.cat([perm_pos, perm_neg])
        train_feats = train_feats[keep]
        train_scores = train_scores[keep]
        train_labels = train_labels[keep]
        print(f"Balanced train: {len(train_labels)} ({(train_labels==1).sum()} pos, {(train_labels==0).sum()} neg)")

    # DataLoader
    train_ds = TensorDataset(train_feats, train_scores, train_labels)
    val_ds = TensorDataset(val_feats, val_scores, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    in_ch = train_feats.shape[1]
    model = ROIScorer(in_channels=in_ch, hidden_dim=args.hidden_dim).to(device)
    print(f"Model: in_ch={in_ch}, params={sum(p.numel() for p in model.parameters())}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for feats_b, scores_b, labels_b in train_loader:
            feats_b = feats_b.to(device)
            scores_b = scores_b.to(device)
            labels_b = labels_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats_b, scores_b)

            # Focal loss (alpha=2, gamma=4)
            bce = F.binary_cross_entropy_with_logits(logits, labels_b, reduction="none")
            prob = logits.detach().sigmoid()
            pt = labels_b * prob + (1 - labels_b) * (1 - prob)
            loss = (2.0 * (1 - pt) ** 4.0 * bce).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Eval
        model.eval()
        tp, fp, tn, fn = 0, 0, 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for feats_b, scores_b, labels_b in val_loader:
                feats_b = feats_b.to(device)
                scores_b = scores_b.to(device)
                labels_b = labels_b.to(device)

                logits = model(feats_b, scores_b)
                bce = F.binary_cross_entropy_with_logits(logits, labels_b, reduction="none")
                prob = logits.sigmoid()
                pt = labels_b * prob + (1 - labels_b) * (1 - prob)
                val_loss += (2.0 * (1 - pt) ** 4.0 * bce).mean().item()

                pred = (prob > 0.5).float()
                tp += ((pred == 1) & (labels_b == 1)).sum().item()
                fp += ((pred == 1) & (labels_b == 0)).sum().item()
                tn += ((pred == 0) & (labels_b == 0)).sum().item()
                fn += ((pred == 0) & (labels_b == 1)).sum().item()

        val_loss /= len(val_loader)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        history.append(record)

        if epoch % 5 == 0 or epoch == 1 or f1 > best_f1:
            print(f"Epoch {epoch}: loss={train_loss:.4f} val_f1={f1:.3f} prec={prec:.3f} rec={rec:.3f}")

        if f1 >= best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": record}, out_dir / "best.pt")

    save_json(out_dir / "best_metrics.json", max(history, key=lambda r: r.get("f1", 0)))
    save_json(out_dir / "history.json", {"history": history})
    print(f"\nBest val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

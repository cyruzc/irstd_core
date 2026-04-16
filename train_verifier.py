"""Train proposal verifier."""
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
from engine import load_checkpoint, save_checkpoint, save_json
from verifier_model import ProposalDataset, ProposalVerifierNet


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        image_patch = batch["image_patch"].to(device, non_blocking=True)
        center_hint = batch["center_hint"].to(device, non_blocking=True)
        point_score = batch["point_score"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image_patch, center_hint, point_score)

        # Focal loss
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, label, reduction="none")
        prob = logits.detach().sigmoid()
        pt = label * prob + (1 - label) * (1 - prob)
        focal_weight = (1 - pt) ** 4.0  # gamma=4.0
        alpha = 2.0
        loss = (alpha * focal_weight * bce).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = (logits.detach().sigmoid() > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.numel()

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    total_loss = 0.0
    count = 0
    for batch in loader:
        image_patch = batch["image_patch"].to(device, non_blocking=True)
        center_hint = batch["center_hint"].to(device, non_blocking=True)
        point_score = batch["point_score"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        logits = model(image_patch, center_hint, point_score)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, label, reduction="none")
        prob = logits.sigmoid()
        pt = label * prob + (1 - label) * (1 - prob)
        loss = (2.0 * (1 - pt) ** 4.0 * bce).mean()
        total_loss += loss.item()
        count += 1

        pred = (logits.sigmoid() > 0.5).float()
        tp += ((pred == 1) & (label == 1)).sum().item()
        fp += ((pred == 1) & (label == 0)).sum().item()
        tn += ((pred == 0) & (label == 0)).sum().item()
        fn += ((pred == 0) & (label == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": total_loss / max(count, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train proposal verifier")
    parser.add_argument("--proposals", type=str, required=True, help="Path to proposals JSON")
    parser.add_argument("--proposals-val", type=str, default=None, help="Path to val proposals JSON (if separate)")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name for normalization params")
    parser.add_argument("--img-mean", type=float, default=0.0, help="Image mean for normalization")
    parser.add_argument("--img-std", type=float, default=1.0, help="Image std for normalization")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--center-hint-sigma", type=float, default=2.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/verifier")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of proposals for validation")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    model = ProposalVerifierNet(
        in_channels=2, base_channels=args.base_channels, hidden_dim=args.hidden_dim,
    ).to(device)

    # Resolve normalization from dataset config if provided
    img_mean = args.img_mean
    img_std = args.img_std
    if args.dataset_name:
        from dataset_config import DATASET_REGISTRY
        cfg = DATASET_REGISTRY[args.dataset_name]
        img_mean = cfg.img_mean
        img_std = cfg.img_std
        print(f"Using normalization from {args.dataset_name}: mean={img_mean}, std={img_std}")

    if args.resume:
        load_checkpoint(Path(args.resume), model, map_location=str(device))
        print(f"Resumed from {args.resume}")

    # If separate val proposals provided, use them; otherwise split
    if args.proposals_val:
        train_set = ProposalDataset(args.proposals, patch_size=args.patch_size, augment=True,
                                    img_mean=img_mean, img_std=img_std)
        val_set = ProposalDataset(args.proposals_val, patch_size=args.patch_size, augment=False,
                                  img_mean=img_mean, img_std=img_std)
    else:
        # Load all, split manually
        import json as _json
        all_data = _json.loads(Path(args.proposals).read_text())
        proposals = [p for p in all_data["proposals"] if p["label"] >= 0]
        # Split by image name to avoid data leakage
        all_names = sorted(set(p["name"] for p in proposals))
        rng = np.random.RandomState(args.seed)
        rng.shuffle(all_names)
        split_idx = int(len(all_names) * (1 - args.val_split))
        train_names = set(all_names[:split_idx])
        val_names = set(all_names[split_idx:])

        # Write temp files
        train_data = {**all_data, "proposals": [p for p in proposals if p["name"] in train_names]}
        val_data = {**all_data, "proposals": [p for p in proposals if p["name"] in val_names]}

        tmp_train = out_dir / "_train_proposals.json"
        tmp_val = out_dir / "_val_proposals.json"
        tmp_train.write_text(_json.dumps(train_data, indent=2), encoding="utf-8")
        tmp_val.write_text(_json.dumps(val_data, indent=2), encoding="utf-8")

        train_set = ProposalDataset(str(tmp_train), patch_size=args.patch_size, augment=True,
                                    img_mean=img_mean, img_std=img_std)
        val_set = ProposalDataset(str(tmp_val), patch_size=args.patch_size, augment=False,
                                  img_mean=img_mean, img_std=img_std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        record = {"epoch": epoch, "train_loss": loss, "train_acc": acc, "lr": float(optimizer.param_groups[0]["lr"]), **val_metrics}
        history.append(record)
        save_json(out_dir / "history.json", {"history": history})

        print(f"Epoch {epoch}: loss={loss:.4f} acc={acc:.4f} | val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f}")

        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(out_dir / "best.pt", model, epoch=epoch, metrics=val_metrics)

    save_json(out_dir / "best_metrics.json", max(history, key=lambda r: r.get("f1", 0)))


if __name__ == "__main__":
    main()

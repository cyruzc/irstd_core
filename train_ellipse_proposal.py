from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_engine import evaluate_full_images, evaluate_instance_level, maybe_save_best, train_one_epoch
from ellipse_losses import EllipseReconstructionLoss
from ellipse_model import CentroidConditionedEllipseNet
from engine import load_checkpoint, save_json
from proposal_ellipse_data import build_proposal_conditioned_dataset


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ellipse reconstructor on matched CNet proposals")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--train-proposals", type=str, required=True)
    parser.add_argument("--val-proposals", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--center-hint-sigma", type=float, default=2.0)
    parser.add_argument("--cache-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--w-dice", type=float, default=1.0)
    parser.add_argument("--w-iou", type=float, default=1.0)
    parser.add_argument("--w-bce", type=float, default=0.5)
    parser.add_argument("--w-moment", type=float, default=0.25)
    parser.add_argument("--w-param", type=float, default=0.25)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)

    train_set = None
    if not args.eval_only:
        train_set = build_proposal_conditioned_dataset(
            config=config,
            proposal_json=args.train_proposals,
            patch_size=args.patch_size,
            center_hint_sigma=args.center_hint_sigma,
            cache_data=args.cache_data,
        )
    val_set = build_proposal_conditioned_dataset(
        config=config,
        proposal_json=args.val_proposals,
        patch_size=args.patch_size,
        center_hint_sigma=args.center_hint_sigma,
        cache_data=args.cache_data,
    )

    device = torch.device(args.device)
    model = CentroidConditionedEllipseNet(
        in_channels=2,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
    ).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model=model, map_location=str(device))
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}.")

    val_loader = make_loader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.eval_only:
        instance_metrics = evaluate_instance_level(model, val_loader, device=device, patch_size=args.patch_size, threshold=args.threshold)
        full_metrics = evaluate_full_images(model, val_loader, device=device, patch_size=args.patch_size, threshold=args.threshold, distance_thresh=args.distance_thresh)
        metrics = {**{f"instance_{k}": v for k, v in instance_metrics.items()}, **{f"full_{k}": v for k, v in full_metrics.items()}}
        save_json(out_dir / "eval_only_metrics.json", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    if train_set is None:
        raise ValueError("Training dataset is required unless --eval-only is set.")
    train_loader = make_loader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = EllipseReconstructionLoss(
        w_dice=args.w_dice,
        w_iou=args.w_iou,
        w_bce=args.w_bce,
        w_moment=args.w_moment,
        w_param=args.w_param,
    )

    history: list[dict[str, float]] = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            patch_size=args.patch_size,
        )
        scheduler.step()

        instance_metrics = evaluate_instance_level(model, val_loader, device=device, patch_size=args.patch_size, threshold=args.threshold)
        full_metrics = evaluate_full_images(model, val_loader, device=device, patch_size=args.patch_size, threshold=args.threshold, distance_thresh=args.distance_thresh)
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"instance_{k}": v for k, v in instance_metrics.items()},
            **{f"full_{k}": v for k, v in full_metrics.items()},
        }
        history.append(record)
        save_json(out_dir / "history.json", {"history": history})
        best_score = maybe_save_best(
            checkpoint_dir=out_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=record,
            best_score=best_score,
            score_key="full_IoU",
        )
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()

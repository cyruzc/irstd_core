from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cde_engine import evaluate_full_images, evaluate_instance_level, maybe_save_best, train_one_epoch
from cde_losses import CanonicalDeformableEllipseLoss
from cde_model import CanonicalDeformableEllipseNet
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_data import build_ellipse_datasets, CentroidConditionedEllipseDataset
from engine import load_checkpoint, save_json


def make_loader(dataset: CentroidConditionedEllipseDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Deformable Ellipse training for IR small targets")
    # dataset
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--image-dir-name", type=str, default=None)
    parser.add_argument("--mask-dir-name", type=str, default=None)
    parser.add_argument("--test-image-dir-name", type=str, default=None)
    parser.add_argument("--test-mask-dir-name", type=str, default=None)

    # data pipeline
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--center-hint-sigma", type=float, default=2.0)
    parser.add_argument("--train-prompt-noise-std", type=float, default=0.5)
    parser.add_argument("--eval-prompt-noise-std", type=float, default=0.0)
    parser.add_argument("--cache-data", action="store_true")

    # model
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # CDE-specific
    parser.add_argument("--num-fourier-terms", type=int, default=3)
    parser.add_argument("--start-k", type=int, default=3)
    parser.add_argument("--deform-scale", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=12.0)
    parser.add_argument("--use-gate", action="store_true")

    # two-stage training
    parser.add_argument("--pretrain-base", type=str, default=None, help="Path to base-only checkpoint for two-stage training")
    parser.add_argument("--freeze-base-epochs", type=int, default=0, help="Freeze backbone+base head for N epochs, then unfreeze")

    # loss weights
    parser.add_argument("--w-dice", type=float, default=1.0)
    parser.add_argument("--w-iou", type=float, default=1.0)
    parser.add_argument("--w-bce", type=float, default=0.5)
    parser.add_argument("--w-moment", type=float, default=0.0)
    parser.add_argument("--w-param", type=float, default=0.25)
    parser.add_argument("--w-fourier", type=float, default=0.02)
    parser.add_argument("--moment-warmup-epochs", type=int, default=0)

    # eval / output
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="runs/cde_reconstruction")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def _load_base_only_checkpoint(model: CanonicalDeformableEllipseNet, path: str, device: torch.device) -> None:
    """Load base-only (num_fourier_terms=0) checkpoint into CDE model.

    Copies encoder + shared head weights, initializes Fourier outputs to zero.
    """
    ckpt = torch.load(path, map_location=str(device), weights_only=False)
    base_state = ckpt["model"]
    model_state = model.state_dict()

    for key, value in base_state.items():
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            model_state[key] = value
        elif "head.4" in key:
            # Last Linear: copy base dims (first 6), zero-init Fourier dims
            if value.dim() == 2:
                model_state[key][:value.shape[0], :] = value
                model_state[key][value.shape[0]:, :] = 0.0
            elif value.dim() == 1:
                model_state[key][:value.shape[0]] = value
                model_state[key][value.shape[0]:] = 0.0

    model.load_state_dict(model_state)
    print(f"Loaded base-only checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")


def _freeze_base(model: CanonicalDeformableEllipseNet) -> None:
    """Freeze encoder + shared head layers (everything except the last Linear's Fourier outputs)."""
    for name, param in model.named_parameters():
        if "head.4" not in name:
            param.requires_grad = False
    # head.4 is the last Linear — keep it fully trainable so Fourier outputs can learn
    print("Froze encoder + shared head. Only last Linear is trainable.")


def _unfreeze_all(model: CanonicalDeformableEllipseNet) -> None:
    for param in model.parameters():
        param.requires_grad = True
    print("Unfroze all parameters.")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=not args.eval_only)

    train_set, val_set = build_ellipse_datasets(
        config=config,
        patch_size=args.patch_size,
        train_prompt_noise_std=args.train_prompt_noise_std,
        eval_prompt_noise_std=args.eval_prompt_noise_std,
        center_hint_sigma=args.center_hint_sigma,
        seed=args.seed,
        cache_data=args.cache_data,
    )

    val_loader = make_loader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    model = CanonicalDeformableEllipseNet(
        in_channels=2,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        num_fourier_terms=args.num_fourier_terms,
        use_gate=args.use_gate,
    ).to(device)

    if args.pretrain_base:
        _load_base_only_checkpoint(model, args.pretrain_base, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model=model, map_location=str(device))
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}.")

    eval_kwargs = dict(
        device=device,
        patch_size=args.patch_size,
        num_fourier_terms=args.num_fourier_terms,
        start_k=args.start_k,
        deform_scale=args.deform_scale,
        temperature=args.temperature,
        use_gate=args.use_gate,
        threshold=args.threshold,
    )

    if args.eval_only:
        instance_metrics = evaluate_instance_level(model, val_loader, **eval_kwargs)
        full_metrics = evaluate_full_images(model, val_loader, **eval_kwargs, distance_thresh=args.distance_thresh)
        metrics = {**{f"instance_{k}": v for k, v in instance_metrics.items()}, **{f"full_{k}": v for k, v in full_metrics.items()}}
        save_json(out_dir / "eval_only_metrics.json", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    if train_set is None:
        raise ValueError("Training split is unavailable for the selected dataset/configuration.")
    train_loader = make_loader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CanonicalDeformableEllipseLoss(
        w_dice=args.w_dice,
        w_iou=args.w_iou,
        w_bce=args.w_bce,
        w_param=args.w_param,
        w_fourier=args.w_fourier,
        w_moment=args.w_moment,
        moment_warmup_epochs=args.moment_warmup_epochs,
        start_k=args.start_k,
    )

    # Two-stage: freeze base for initial epochs
    base_frozen = False
    if args.freeze_base_epochs > 0:
        _freeze_base(model)
        base_frozen = True
        # Rebuild optimizer with only trainable params
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float]] = []
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze base after freeze_base_epochs
        if base_frozen and epoch > args.freeze_base_epochs:
            _unfreeze_all(model)
            base_frozen = False
            # Rebuild optimizer with all params, smaller lr for fine-tuning
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)
            print(f"Epoch {epoch}: switched to joint fine-tuning (lr={args.lr * 0.1:.6f})")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            patch_size=args.patch_size,
            num_fourier_terms=args.num_fourier_terms,
            start_k=args.start_k,
            deform_scale=args.deform_scale,
            temperature=args.temperature,
            use_gate=args.use_gate,
            current_epoch=epoch,
        )
        scheduler.step()

        instance_metrics = evaluate_instance_level(model, val_loader, **eval_kwargs)
        full_metrics = evaluate_full_images(model, val_loader, **eval_kwargs, distance_thresh=args.distance_thresh)
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

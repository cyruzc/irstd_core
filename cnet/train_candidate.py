from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from candidate_loss import CandidateFormationLoss
from candidate_model import CandidateFormationNet
from candidate_utils import extract_candidate_centers, match_candidates
from data import PointSupervisionDataset, worker_init_fn
from dataset_config import build_dataset_config, validate_dataset_config
from engine import save_checkpoint, save_json


def build_thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-search-step must be > 0.")
    thresholds: list[float] = []
    value = start
    while value <= stop + 1e-8:
        thresholds.append(round(value, 6))
        value += step
    return thresholds


@torch.no_grad()
def evaluate_candidate_epoch(
    model: CandidateFormationNet,
    loader: DataLoader,
    device: torch.device,
    distance_thresh: float = 3.0,
    search_thresholds: list[float] | None = None,
) -> dict[str, float]:
    model.eval()
    thresholds = search_thresholds or [model.score_threshold]
    stats = {
        threshold: {"matched": 0, "pred": 0, "gt": 0}
        for threshold in thresholds
    }

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        point = batch["point"].to(device, non_blocking=True)
        outputs = model(image, return_candidates=False)
        proposal_map = outputs["proposal_map"]

        for threshold in thresholds:
            candidates = extract_candidate_centers(
                proposal_map,
                topk=model.max_candidates,
                nms_kernel=model.nms_kernel,
                score_threshold=threshold,
            )
            cur_matched, cur_pred, cur_gt = match_candidates(
                candidates["coords"], candidates["valid_mask"], point, distance_thresh=distance_thresh
            )
            stats[threshold]["matched"] += cur_matched
            stats[threshold]["pred"] += cur_pred
            stats[threshold]["gt"] += cur_gt

    best_metrics: dict[str, float] | None = None
    for threshold in thresholds:
        matched = stats[threshold]["matched"]
        total_pred = stats[threshold]["pred"]
        total_gt = stats[threshold]["gt"]
        precision = matched / max(total_pred, 1)
        recall = matched / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Matched": float(matched),
            "Predictions": float(total_pred),
            "Targets": float(total_gt),
            "BestThreshold": float(threshold),
        }
        if best_metrics is None or metrics["F1"] > best_metrics["F1"]:
            best_metrics = metrics

    assert best_metrics is not None
    return best_metrics


def train_one_epoch(
    model: CandidateFormationNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CandidateFormationLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        point = batch["point"].to(device, non_blocking=True)
        batch = {**batch, "point": point}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, return_candidates=True)
        loss = criterion(outputs, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())

    metrics = {"loss": total_loss / max(len(loader), 1)}
    metrics.update(criterion.last_components)
    return metrics


def build_loader(dataset: PointSupervisionDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an independent candidate formation network.")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--nms-kernel", type=int, default=7)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--ring-inner-radius", type=int, default=2)
    parser.add_argument("--ring-outer-radius", type=int, default=6)
    parser.add_argument("--vote-radius", type=int, default=5)
    parser.add_argument("--peak-ranking-weight", type=float, default=0.2)
    parser.add_argument("--peak-ranking-margin", type=float, default=0.15)
    parser.add_argument("--over-count-weight", type=float, default=1.0)
    parser.add_argument("--under-count-weight", type=float, default=0.25)
    parser.add_argument("--redundancy-radius", type=float, default=6.0)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--threshold-search-start", type=float, default=0.10)
    parser.add_argument("--threshold-search-stop", type=float, default=0.80)
    parser.add_argument("--threshold-search-step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = build_dataset_config(args)
    validate_dataset_config(config)
    if config.centroid_label_dir in (None, ""):
        raise ValueError("Candidate formation training requires centroid point labels.")

    train_set = PointSupervisionDataset(
        dataset_root=config.root,
        split_name=config.train_split,
        point_label_dir=config.centroid_label_dir,
        image_dir_name=config.train_image_dir,
        mask_dir_name=config.train_mask_dir,
        train=True,
        patch_size=args.patch_size,
        seed=args.seed,
        cache_data=args.cache_data,
        img_mean=config.img_mean,
        img_std=config.img_std,
    )
    val_set = PointSupervisionDataset(
        dataset_root=config.root,
        split_name=config.test_split,
        point_label_dir=config.centroid_label_dir,
        image_dir_name=config.test_image_dir,
        mask_dir_name=config.test_mask_dir,
        train=False,
        seed=args.seed,
        cache_data=args.cache_data,
        img_mean=config.img_mean,
        img_std=config.img_std,
    )

    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)

    model = CandidateFormationNet(
        in_channels=1,
        base_channels=args.base_channels,
        max_candidates=args.max_candidates,
        nms_kernel=args.nms_kernel,
        score_threshold=args.score_threshold,
        vote_radius=args.vote_radius,
    ).to(device)
    criterion = CandidateFormationLoss(
        sigma=args.gaussian_sigma,
        ring_inner_radius=args.ring_inner_radius,
        ring_outer_radius=args.ring_outer_radius,
        vote_radius=args.vote_radius,
        peak_ranking_weight=args.peak_ranking_weight,
        peak_ranking_margin=args.peak_ranking_margin,
        over_count_weight=args.over_count_weight,
        under_count_weight=args.under_count_weight,
        redundancy_radius=args.redundancy_radius,
        topk=args.max_candidates,
        nms_kernel=args.nms_kernel,
        score_threshold=args.score_threshold,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    threshold_candidates = build_thresholds(
        args.threshold_search_start,
        args.threshold_search_stop,
        args.threshold_search_step,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_candidate_epoch(
            model,
            val_loader,
            device,
            distance_thresh=args.distance_thresh,
            search_thresholds=threshold_candidates,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_metrics = {"epoch": float(epoch), "lr": current_lr, **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        save_json(output_dir / "history.json", {"epochs": history})

        if val_metrics["F1"] > best_f1:
            best_f1 = val_metrics["F1"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                epoch=epoch,
                metrics=epoch_metrics,
                optimizer=optimizer,
                extra={"config": vars(args), "scheduler": scheduler.state_dict()},
            )

        print(
            f"epoch={epoch:03d} "
            f"lr={current_lr:.7f} "
            f"loss={train_metrics['loss']:.4f} "
            f"f1={val_metrics['F1']:.4f} "
            f"precision={val_metrics['Precision']:.4f} "
            f"recall={val_metrics['Recall']:.4f} "
            f"best_threshold={val_metrics['BestThreshold']:.2f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()

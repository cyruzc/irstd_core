from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_data import build_ellipse_dataset
from ellipse_engine import evaluate_full_images, evaluate_instance_level
from ellipse_model import CentroidConditionedEllipseNet
from engine import load_checkpoint


def parse_noise_values(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("At least one noise std is required.")
    return values


def make_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ellipse checkpoint under different prompt noise levels")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--test-image-dir-name", type=str, default=None)
    parser.add_argument("--test-mask-dir-name", type=str, default=None)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--center-hint-sigma", type=float, default=2.0)
    parser.add_argument("--noise-stds", type=str, default="0,0.5,1,1.5,2,3,4")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)
    noise_values = parse_noise_values(args.noise_stds)
    device = torch.device(args.device)

    model = CentroidConditionedEllipseNet(
        in_channels=2,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
    ).to(device)
    load_checkpoint(Path(args.resume), model=model, map_location=str(device))
    model.eval()

    results: list[dict[str, float]] = []
    for noise_std in noise_values:
        val_set = build_ellipse_dataset(
            config=config,
            split_name=config.test_split,
            image_dir_name=config.test_image_dir,
            mask_dir_name=config.test_mask_dir,
            patch_size=args.patch_size,
            prompt_noise_std=noise_std,
            center_hint_sigma=args.center_hint_sigma,
            seed=42,
            cache_data=False,
        )
        loader = make_loader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
        instance_metrics = evaluate_instance_level(
            model,
            loader,
            device=device,
            patch_size=args.patch_size,
            threshold=args.threshold,
        )
        full_metrics = evaluate_full_images(
            model,
            loader,
            device=device,
            patch_size=args.patch_size,
            threshold=args.threshold,
            distance_thresh=args.distance_thresh,
        )
        record = {
            "noise_std": noise_std,
            **{f"instance_{k}": v for k, v in instance_metrics.items()},
            **{f"full_{k}": v for k, v in full_metrics.items()},
        }
        results.append(record)
        print(json.dumps(record, sort_keys=True))

    payload = {
        "config": {
            "dataset_name": config.name,
            "resume": args.resume,
            "patch_size": args.patch_size,
            "center_hint_sigma": args.center_hint_sigma,
            "noise_stds": noise_values,
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

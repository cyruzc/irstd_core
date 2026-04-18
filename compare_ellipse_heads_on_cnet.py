from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cnet_proposal_utils import gt_instances_from_mask, load_cnet_from_checkpoint, nearest_instance_assignment, resolve_records_for_split, run_cnet_on_record
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_model import CentroidConditionedEllipseNet
from ellipse_renderer import decode_raw_params, raw_to_soft_mask
from ellipse_utils import crop_with_pad, make_center_hint
from engine import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ellipse checkpoints on the same CNet proposals")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--use-test-dirs", action="store_true")
    parser.add_argument("--cnet-checkpoint", type=str, required=True)
    parser.add_argument("--old-ellipse-checkpoint", type=str, required=True)
    parser.add_argument("--new-ellipse-checkpoint", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--center-hint-sigma", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def init_model(ckpt_path: str, device: torch.device, base_channels: int, hidden_dim: int) -> CentroidConditionedEllipseNet:
    model = CentroidConditionedEllipseNet(in_channels=2, base_channels=base_channels, hidden_dim=hidden_dim).to(device)
    load_checkpoint(Path(ckpt_path), model=model, map_location=str(device))
    model.eval()
    return model


def collect_stats(bucket: defaultdict[str, list[float]], prefix: str, decoded: dict[str, torch.Tensor], pred_patch: np.ndarray, gt_patch: np.ndarray) -> None:
    area = float(pred_patch.sum())
    gt_area = float(gt_patch.sum())
    bucket[f"{prefix}_area"].append(area)
    bucket[f"{prefix}_gt_area"].append(gt_area)
    bucket[f"{prefix}_area_ratio"].append(area / max(gt_area, 1.0))
    bucket[f"{prefix}_dx_abs"].append(float(decoded["dx"][0].detach().abs()))
    bucket[f"{prefix}_dy_abs"].append(float(decoded["dy"][0].detach().abs()))
    bucket[f"{prefix}_a"].append(float(decoded["a"][0].detach()))
    bucket[f"{prefix}_b"].append(float(decoded["b"][0].detach()))
    bucket[f"{prefix}_aspect"].append(float(decoded["a"][0].detach() / max(float(decoded["b"][0].detach()), 1e-3)))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)
    split_name = args.split or config.test_split
    use_test_dirs = args.use_test_dirs or split_name == config.test_split

    cnet_model, _ = load_cnet_from_checkpoint(args.cnet_checkpoint, device)
    old_model = init_model(args.old_ellipse_checkpoint, device, args.base_channels, args.hidden_dim)
    new_model = init_model(args.new_ellipse_checkpoint, device, args.base_channels, args.hidden_dim)
    records = resolve_records_for_split(config, split_name=split_name, use_test_dirs=use_test_dirs)
    center_hint = make_center_hint(args.patch_size, sigma=args.center_hint_sigma)

    overall: defaultdict[str, list[float]] = defaultdict(list)
    matched: defaultdict[str, list[float]] = defaultdict(list)
    unmatched: defaultdict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for record in records:
            image_norm, proposals = run_cnet_on_record(
                cnet_model,
                image_path=record.image_path,
                img_mean=config.img_mean,
                img_std=config.img_std,
                device=device,
            )
            instances = gt_instances_from_mask(record.mask_path)

            for proposal in proposals:
                inst_id, dist = nearest_instance_assignment(proposal["x"], proposal["y"], instances)
                is_matched = inst_id is not None and dist <= args.match_radius
                if inst_id is not None:
                    gt_comp = instances[inst_id]["mask"]
                else:
                    gt_comp = np.zeros_like((image_norm > 0).astype(np.float32))

                image_patch, _ = crop_with_pad(image_norm, proposal["x"], proposal["y"], args.patch_size, pad_value=0.0)
                gt_patch, _ = crop_with_pad(gt_comp, proposal["x"], proposal["y"], args.patch_size, pad_value=0.0)

                image_t = torch.from_numpy(image_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                hint_t = torch.from_numpy(center_hint).unsqueeze(0).unsqueeze(0).float().to(device)

                old_raw = old_model(image_t, hint_t)
                new_raw = new_model(image_t, hint_t)
                old_mask, old_decoded = raw_to_soft_mask(old_raw, patch_size=args.patch_size)
                new_mask, new_decoded = raw_to_soft_mask(new_raw, patch_size=args.patch_size)
                old_patch = (old_mask[0, 0].detach().cpu().numpy() > args.threshold).astype(np.float32)
                new_patch = (new_mask[0, 0].detach().cpu().numpy() > args.threshold).astype(np.float32)

                target = matched if is_matched else unmatched
                for bucket in (overall, target):
                    bucket["proposal_score"].append(float(proposal["score"]))
                    bucket["proposal_distance"].append(float(dist))
                    collect_stats(bucket, "old", old_decoded, old_patch, gt_patch)
                    collect_stats(bucket, "new", new_decoded, new_patch, gt_patch)
                    bucket["delta_area"].append(float(new_patch.sum() - old_patch.sum()))
                    bucket["delta_area_ratio"].append(float((new_patch.sum() / max(gt_patch.sum(), 1.0)) - (old_patch.sum() / max(gt_patch.sum(), 1.0))))
                    bucket["delta_dx_abs"].append(float(new_decoded["dx"][0].detach().abs() - old_decoded["dx"][0].detach().abs()))
                    bucket["delta_dy_abs"].append(float(new_decoded["dy"][0].detach().abs() - old_decoded["dy"][0].detach().abs()))
                    bucket["delta_a"].append(float(new_decoded["a"][0].detach() - old_decoded["a"][0].detach()))
                    bucket["delta_b"].append(float(new_decoded["b"][0].detach() - old_decoded["b"][0].detach()))

    payload = {
        "config": {
            "dataset_name": config.name,
            "split": split_name,
            "cnet_checkpoint": args.cnet_checkpoint,
            "old_ellipse_checkpoint": args.old_ellipse_checkpoint,
            "new_ellipse_checkpoint": args.new_ellipse_checkpoint,
            "match_radius": args.match_radius,
            "threshold": args.threshold,
        },
        "overall": {k: summarize(v) for k, v in sorted(overall.items())},
        "matched": {k: summarize(v) for k, v in sorted(matched.items())},
        "unmatched": {k: summarize(v) for k, v in sorted(unmatched.items())},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved comparison to {args.output}")


if __name__ == "__main__":
    main()

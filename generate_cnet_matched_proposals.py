from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cnet_proposal_utils import (
    gt_instances_from_mask,
    load_cnet_from_checkpoint,
    nearest_instance_assignment,
    resolve_records_for_split,
    run_cnet_on_record,
    save_json,
)
from dataset_config import build_dataset_config, validate_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matched CNet proposals for proposal-conditioned ellipse training")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--use-test-dirs", action="store_true")
    parser.add_argument("--cnet-checkpoint", type=str, required=True)
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--selection", type=str, default="nearest", choices=["nearest", "highest_score", "all"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)
    split_name = args.split or config.train_split
    if split_name in (None, ""):
        raise ValueError("A split is required.")
    use_test_dirs = args.use_test_dirs or split_name == config.test_split

    model, ckpt = load_cnet_from_checkpoint(args.cnet_checkpoint, device)
    records = resolve_records_for_split(config, split_name=split_name, use_test_dirs=use_test_dirs)

    matched_entries: list[dict] = []
    per_instance_pool: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
    total_gt = 0
    matched_gt = 0
    total_proposals = 0
    total_matched = 0

    for record in records:
        _, proposals = run_cnet_on_record(
            model,
            image_path=record.image_path,
            img_mean=config.img_mean,
            img_std=config.img_std,
            device=device,
        )
        instances = gt_instances_from_mask(record.mask_path)
        total_gt += len(instances)
        total_proposals += len(proposals)
        instance_by_id = {int(inst["instance_id"]): inst for inst in instances}

        seen_gt: set[int] = set()
        for proposal in proposals:
            inst_id, dist = nearest_instance_assignment(proposal["x"], proposal["y"], instances)
            if inst_id is None or dist > args.match_radius:
                continue
            total_matched += 1
            seen_gt.add(inst_id)
            inst = instance_by_id[inst_id]
            entry = {
                "name": record.name,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "instance_id": int(inst_id),
                "prompt_x": float(proposal["x"]),
                "prompt_y": float(proposal["y"]),
                "gt_x": float(inst["centroid_x"]),
                "gt_y": float(inst["centroid_y"]),
                "distance_to_gt": float(dist),
                "score": float(proposal["score"]),
                "area": int(inst["area"]),
                "bbox": [int(v) for v in inst["bbox"]],
            }
            if args.selection == "all":
                matched_entries.append(entry)
            else:
                per_instance_pool[(record.name, int(inst_id))].append(entry)
        matched_gt += len(seen_gt)

    if args.selection != "all":
        for key, pool in per_instance_pool.items():
            if args.selection == "nearest":
                chosen = min(pool, key=lambda x: (x["distance_to_gt"], -x["score"]))
            else:
                chosen = max(pool, key=lambda x: (x["score"], -x["distance_to_gt"]))
            matched_entries.append(chosen)

    distances = [entry["distance_to_gt"] for entry in matched_entries]
    payload = {
        "config": {
            "dataset_name": config.name,
            "split": split_name,
            "use_test_dirs": use_test_dirs,
            "cnet_checkpoint": args.cnet_checkpoint,
            "match_radius": args.match_radius,
            "selection": args.selection,
            "checkpoint_config": ckpt.get("config", {}),
        },
        "stats": {
            "total_images": len(records),
            "total_gt": total_gt,
            "matched_gt_within_radius": matched_gt,
            "gt_recall_within_radius": float(matched_gt / max(total_gt, 1)),
            "total_proposals": total_proposals,
            "total_matched_proposals": total_matched,
            "selected_entries": len(matched_entries),
            "mean_distance_to_gt": float(np.mean(distances)) if distances else 0.0,
            "median_distance_to_gt": float(np.median(distances)) if distances else 0.0,
            "p90_distance_to_gt": float(np.percentile(distances, 90)) if distances else 0.0,
        },
        "entries": sorted(matched_entries, key=lambda x: (x["name"], x["instance_id"], x["distance_to_gt"])),
    }
    save_json(args.output, payload)
    print(json.dumps(payload["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

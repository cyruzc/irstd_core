from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Analyze CNet proposal error distribution against GT centroids")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, help="Override split file")
    parser.add_argument("--use-test-dirs", action="store_true", help="Use test image/mask dirs with the provided split")
    parser.add_argument("--cnet-checkpoint", type=str, required=True)
    parser.add_argument("--match-radius", type=float, default=3.0)
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)
    split_name = args.split or config.test_split
    use_test_dirs = args.use_test_dirs or split_name == config.test_split

    model, ckpt = load_cnet_from_checkpoint(args.cnet_checkpoint, device)
    records = resolve_records_for_split(config, split_name=split_name, use_test_dirs=use_test_dirs)

    proposal_distances: list[float] = []
    matched_distances: list[float] = []
    gt_best_distances: list[float] = []
    dx_values: list[float] = []
    dy_values: list[float] = []
    matched_scores: list[float] = []
    unmatched_scores: list[float] = []
    proposal_count_per_image: list[int] = []
    matched_count_per_image: list[int] = []
    bucket_edges = [0.5, 1.0, 2.0, 3.0, 5.0]
    distance_buckets: defaultdict[str, int] = defaultdict(int)
    per_image_samples: list[dict] = []

    total_gt = 0
    total_matched_proposals = 0
    total_unmatched_proposals = 0
    total_matched_gt = 0

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
        proposal_count_per_image.append(len(proposals))

        gt_best = {int(inst["instance_id"]): float("inf") for inst in instances}
        image_matched = 0
        for proposal in proposals:
            inst_id, dist = nearest_instance_assignment(proposal["x"], proposal["y"], instances)
            proposal_distances.append(dist)
            if inst_id is not None:
                inst = instances[inst_id]
                dx = float(proposal["x"] - float(inst["centroid_x"]))
                dy = float(proposal["y"] - float(inst["centroid_y"]))
                dx_values.append(dx)
                dy_values.append(dy)
                gt_best[inst_id] = min(gt_best[inst_id], dist)
            if dist <= args.match_radius:
                matched_distances.append(dist)
                matched_scores.append(float(proposal["score"]))
                total_matched_proposals += 1
                image_matched += 1
                if dist <= bucket_edges[0]:
                    distance_buckets["<=0.5"] += 1
                elif dist <= bucket_edges[1]:
                    distance_buckets["0.5-1"] += 1
                elif dist <= bucket_edges[2]:
                    distance_buckets["1-2"] += 1
                elif dist <= bucket_edges[3]:
                    distance_buckets["2-3"] += 1
                elif dist <= bucket_edges[4]:
                    distance_buckets["3-5"] += 1
                else:
                    distance_buckets[">5"] += 1
            else:
                unmatched_scores.append(float(proposal["score"]))
                total_unmatched_proposals += 1

        matched_count_per_image.append(image_matched)
        matched_gt_ids = 0
        for best in gt_best.values():
            if np.isfinite(best):
                gt_best_distances.append(best)
                if best <= args.match_radius:
                    matched_gt_ids += 1
        total_matched_gt += matched_gt_ids
        per_image_samples.append(
            {
                "name": record.name,
                "num_gt": len(instances),
                "num_proposals": len(proposals),
                "num_matched_proposals": image_matched,
                "num_matched_gt": matched_gt_ids,
            }
        )

    payload = {
        "config": {
            "dataset_name": config.name,
            "split": split_name,
            "use_test_dirs": use_test_dirs,
            "cnet_checkpoint": args.cnet_checkpoint,
            "match_radius": args.match_radius,
            "checkpoint_config": ckpt.get("config", {}),
        },
        "summary": {
            "num_images": len(records),
            "total_gt": total_gt,
            "total_matched_gt": total_matched_gt,
            "gt_recall_at_radius": float(total_matched_gt / max(total_gt, 1)),
            "total_proposals": total_matched_proposals + total_unmatched_proposals,
            "total_matched_proposals": total_matched_proposals,
            "total_unmatched_proposals": total_unmatched_proposals,
            "proposal_precision_at_radius": float(total_matched_proposals / max(total_matched_proposals + total_unmatched_proposals, 1)),
        },
        "proposal_distance_to_nearest_gt": summarize(proposal_distances),
        "matched_proposal_distance": summarize(matched_distances),
        "gt_best_distance_to_any_proposal": summarize(gt_best_distances),
        "dx_distribution": summarize(dx_values),
        "dy_distribution": summarize(dy_values),
        "proposal_score_matched": summarize(matched_scores),
        "proposal_score_unmatched": summarize(unmatched_scores),
        "proposal_count_per_image": summarize(proposal_count_per_image),
        "matched_proposal_count_per_image": summarize(matched_count_per_image),
        "matched_distance_buckets": dict(sorted(distance_buckets.items())),
        "per_image_samples_head": per_image_samples[:50],
    }
    save_json(args.output, payload)
    print(f"saved analysis to {args.output}")


if __name__ == "__main__":
    main()

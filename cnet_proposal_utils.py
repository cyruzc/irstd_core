from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cnet_model import CandidateFormationNet
from data import load_grayscale, resolve_full_records
from dataset_config import DatasetConfig
from ellipse_utils import component_instances


def load_cnet_from_checkpoint(path: str | Path, device: torch.device) -> tuple[CandidateFormationNet, dict]:
    ckpt_path = Path(path)
    ckpt = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    cfg = ckpt.get("config", {})
    model = CandidateFormationNet(
        in_channels=1,
        base_channels=int(cfg.get("base_channels", 32)),
        max_candidates=int(cfg.get("max_candidates", 16)),
        nms_kernel=int(cfg.get("nms_kernel", 7)),
        score_threshold=float(cfg.get("score_threshold", 0.25)),
        vote_radius=float(cfg.get("vote_radius", 5.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_cnet_on_record(
    model: CandidateFormationNet,
    image_path: Path,
    img_mean: float,
    img_std: float,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    image = load_grayscale(image_path)
    image_norm = ((image - img_mean) / img_std).astype(np.float32)
    image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    outputs = model(image_tensor, return_candidates=True)
    coords = outputs["coords"][0]
    scores = outputs["scores"][0]
    valid = outputs["valid_mask"][0]
    proposals: list[dict[str, float]] = []
    for i in range(coords.shape[0]):
        if not bool(valid[i]):
            continue
        cy, cx = float(coords[i, 0]), float(coords[i, 1])
        proposals.append({
            "x": float(cx),
            "y": float(cy),
            "score": float(scores[i]),
        })
    return image_norm, proposals


def gt_instances_from_mask(mask_path: Path) -> list[dict]:
    mask = (load_grayscale(mask_path) > 0).astype(np.float32)
    return component_instances(mask)


def resolve_records_for_split(config: DatasetConfig, split_name: str, use_test_dirs: bool) -> list:
    image_dir = config.test_image_dir if use_test_dirs else config.train_image_dir
    mask_dir = config.test_mask_dir if use_test_dirs else config.train_mask_dir
    return resolve_full_records(
        config.root,
        split_name,
        image_dir_name=image_dir,
        mask_dir_name=mask_dir,
    )


def nearest_instance_assignment(proposal_x: float, proposal_y: float, instances: list[dict]) -> tuple[int | None, float]:
    if not instances:
        return None, float("inf")
    best_idx = None
    best_dist = float("inf")
    for inst in instances:
        dx = proposal_x - float(inst["centroid_x"])
        dy = proposal_y - float(inst["centroid_y"])
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dist < best_dist:
            best_dist = dist
            best_idx = int(inst["instance_id"])
    return best_idx, best_dist


def save_json(path: str | Path, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

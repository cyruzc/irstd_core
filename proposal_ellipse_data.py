from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import load_grayscale
from dataset_config import DatasetConfig
from ellipse_utils import crop_with_pad, fit_ellipse_from_mask, make_center_hint


@dataclass(frozen=True)
class ProposalEllipseEntry:
    name: str
    image_path: Path
    mask_path: Path
    instance_id: int
    prompt_x: float
    prompt_y: float
    gt_x: float
    gt_y: float
    score: float
    distance_to_gt: float


class ProposalConditionedEllipseDataset(Dataset):
    def __init__(
        self,
        proposal_json: str | Path,
        patch_size: int = 32,
        center_hint_sigma: float = 2.0,
        img_mean: float = 0.0,
        img_std: float = 1.0,
        cache_data: bool = False,
    ) -> None:
        self.proposal_json = Path(proposal_json)
        payload = json.loads(self.proposal_json.read_text(encoding="utf-8"))
        raw_entries = payload["entries"]
        self.patch_size = int(patch_size)
        self.center_hint_sigma = float(center_hint_sigma)
        self.img_mean = float(img_mean)
        self.img_std = float(img_std)
        self.cache_data = bool(cache_data)

        self.entries: list[ProposalEllipseEntry] = [
            ProposalEllipseEntry(
                name=str(item["name"]),
                image_path=Path(item["image_path"]),
                mask_path=Path(item["mask_path"]),
                instance_id=int(item["instance_id"]),
                prompt_x=float(item["prompt_x"]),
                prompt_y=float(item["prompt_y"]),
                gt_x=float(item["gt_x"]),
                gt_y=float(item["gt_y"]),
                score=float(item.get("score", 0.0)),
                distance_to_gt=float(item["distance_to_gt"]),
            )
            for item in raw_entries
        ]
        self.cached_images: dict[str, np.ndarray] = {}
        self.cached_masks: dict[str, np.ndarray] = {}
        self.instance_masks: dict[tuple[str, int], np.ndarray] = {}

        for entry in self.entries:
            key = (entry.name, entry.instance_id)
            if key in self.instance_masks:
                continue
            mask = (load_grayscale(entry.mask_path) > 0).astype(np.float32)
            labeled = self._component_instances(mask)
            try:
                self.instance_masks[key] = labeled[entry.instance_id]
            except KeyError as exc:
                raise KeyError(f"Instance {key} missing in {entry.mask_path}") from exc
            if self.cache_data and entry.name not in self.cached_images:
                self.cached_images[entry.name] = load_grayscale(entry.image_path)
                self.cached_masks[entry.name] = mask

        if not self.entries:
            raise RuntimeError(f"No proposal entries found in {self.proposal_json}")

    @staticmethod
    def _component_instances(mask: np.ndarray) -> dict[int, np.ndarray]:
        from ellipse_utils import component_instances

        return {int(inst["instance_id"]): inst["mask"] for inst in component_instances(mask)}

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, entry: ProposalEllipseEntry) -> np.ndarray:
        if entry.name in self.cached_images:
            return self.cached_images[entry.name]
        return load_grayscale(entry.image_path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        image = self._load_image(entry)
        component_mask = self.instance_masks[(entry.name, entry.instance_id)]

        image_patch, crop_meta = crop_with_pad(image, entry.prompt_x, entry.prompt_y, self.patch_size, pad_value=self.img_mean)
        mask_patch, _ = crop_with_pad(component_mask, entry.prompt_x, entry.prompt_y, self.patch_size, pad_value=0.0)

        image_patch = (image_patch.astype(np.float32) - self.img_mean) / self.img_std
        mask_patch = (mask_patch > 0).astype(np.float32)
        center_hint = make_center_hint(self.patch_size, sigma=self.center_hint_sigma)

        ellipse = fit_ellipse_from_mask(mask_patch)
        patch_center = (self.patch_size - 1) / 2.0
        gt_params = np.array(
            [
                ellipse.cx - patch_center,
                ellipse.cy - patch_center,
                ellipse.a,
                ellipse.b,
                ellipse.phi,
            ],
            dtype=np.float32,
        )

        return {
            "name": entry.name,
            "instance_id": torch.tensor(entry.instance_id, dtype=torch.long),
            "image": torch.from_numpy(image_patch).unsqueeze(0).float(),
            "center_hint": torch.from_numpy(center_hint).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_patch).unsqueeze(0).float(),
            "gt_params": torch.from_numpy(gt_params).float(),
            "prompt_x": torch.tensor(entry.prompt_x, dtype=torch.float32),
            "prompt_y": torch.tensor(entry.prompt_y, dtype=torch.float32),
            "gt_x": torch.tensor(entry.gt_x, dtype=torch.float32),
            "gt_y": torch.tensor(entry.gt_y, dtype=torch.float32),
            "proposal_score": torch.tensor(entry.score, dtype=torch.float32),
            "proposal_distance": torch.tensor(entry.distance_to_gt, dtype=torch.float32),
            "image_h": torch.tensor(int(image.shape[0]), dtype=torch.long),
            "image_w": torch.tensor(int(image.shape[1]), dtype=torch.long),
            "src_top": torch.tensor(int(crop_meta.src_top), dtype=torch.long),
            "src_left": torch.tensor(int(crop_meta.src_left), dtype=torch.long),
            "src_bottom": torch.tensor(int(crop_meta.src_bottom), dtype=torch.long),
            "src_right": torch.tensor(int(crop_meta.src_right), dtype=torch.long),
            "dst_top": torch.tensor(int(crop_meta.dst_top), dtype=torch.long),
            "dst_left": torch.tensor(int(crop_meta.dst_left), dtype=torch.long),
            "patch_size": torch.tensor(int(crop_meta.patch_size), dtype=torch.long),
        }


def build_proposal_conditioned_dataset(
    config: DatasetConfig,
    proposal_json: str | Path,
    patch_size: int,
    center_hint_sigma: float,
    cache_data: bool = False,
) -> ProposalConditionedEllipseDataset:
    return ProposalConditionedEllipseDataset(
        proposal_json=proposal_json,
        patch_size=patch_size,
        center_hint_sigma=center_hint_sigma,
        img_mean=config.img_mean,
        img_std=config.img_std,
        cache_data=cache_data,
    )


__all__ = ["ProposalConditionedEllipseDataset", "build_proposal_conditioned_dataset"]

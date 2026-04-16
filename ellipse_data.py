from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import load_grayscale, resolve_full_records
from dataset_config import DatasetConfig
from ellipse_utils import component_instances, crop_with_pad, fit_ellipse_from_mask, make_center_hint, sample_prompt_center, InstanceRecord


@dataclass(frozen=True)
class EllipseSampleMeta:
    name: str
    instance_id: int
    prompt_x: float
    prompt_y: float
    gt_x: float
    gt_y: float
    image_h: int
    image_w: int
    src_top: int
    src_left: int
    src_bottom: int
    src_right: int
    dst_top: int
    dst_left: int
    patch_size: int


class CentroidConditionedEllipseDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split_name: str,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        patch_size: int = 32,
        prompt_noise_std: float = 0.0,
        center_hint_sigma: float = 2.0,
        img_mean: float = 0.0,
        img_std: float = 1.0,
        seed: int = 42,
        min_area: int = 1,
        cache_data: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.patch_size = int(patch_size)
        self.prompt_noise_std = float(prompt_noise_std)
        self.center_hint_sigma = float(center_hint_sigma)
        self.img_mean = float(img_mean)
        self.img_std = float(img_std)
        self._seed = int(seed)
        self.cache_data = cache_data

        full_records = resolve_full_records(
            self.dataset_root,
            split_name,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
        )
        self.records: list[InstanceRecord] = []
        self.cached_images: dict[str, np.ndarray] = {}
        self.cached_masks: dict[str, np.ndarray] = {}
        self._instance_masks: dict[tuple[str, int], np.ndarray] = {}

        for full_record in full_records:
            mask = (load_grayscale(full_record.mask_path) > 0).astype(np.float32)
            for inst in component_instances(mask):
                if inst["area"] < min_area:
                    continue
                iid = int(inst["instance_id"])
                self.records.append(
                    InstanceRecord(
                        name=full_record.name,
                        image_path=full_record.image_path,
                        mask_path=full_record.mask_path,
                        instance_id=iid,
                        centroid_x=float(inst["centroid_x"]),
                        centroid_y=float(inst["centroid_y"]),
                        area=int(inst["area"]),
                        bbox=tuple(inst["bbox"]),
                    )
                )
                self._instance_masks[(full_record.name, iid)] = inst["mask"]
            if cache_data:
                self.cached_images[full_record.name] = load_grayscale(full_record.image_path)
                self.cached_masks[full_record.name] = mask

        if not self.records:
            raise RuntimeError("No valid target instances were found in the requested split.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_mask(self, record: InstanceRecord) -> tuple[np.ndarray, np.ndarray]:
        if record.name in self.cached_images:
            return self.cached_images[record.name], self.cached_masks[record.name]
        image = load_grayscale(record.image_path)
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image, _ = self._load_image_mask(record)
        component_mask = self._instance_masks[(record.name, record.instance_id)]

        rng = np.random.default_rng(self._seed + index)
        prompt_x, prompt_y = sample_prompt_center(record.centroid_x, record.centroid_y, self.prompt_noise_std, rng)
        image_patch, crop_meta = crop_with_pad(image, prompt_x, prompt_y, self.patch_size, pad_value=self.img_mean)
        mask_patch, _ = crop_with_pad(component_mask, prompt_x, prompt_y, self.patch_size, pad_value=0.0)

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

        meta = EllipseSampleMeta(
            name=record.name,
            instance_id=record.instance_id,
            prompt_x=float(prompt_x),
            prompt_y=float(prompt_y),
            gt_x=float(record.centroid_x),
            gt_y=float(record.centroid_y),
            image_h=int(image.shape[0]),
            image_w=int(image.shape[1]),
            src_top=int(crop_meta.src_top),
            src_left=int(crop_meta.src_left),
            src_bottom=int(crop_meta.src_bottom),
            src_right=int(crop_meta.src_right),
            dst_top=int(crop_meta.dst_top),
            dst_left=int(crop_meta.dst_left),
            patch_size=int(crop_meta.patch_size),
        )

        return {
            "name": record.name,
            "instance_id": torch.tensor(record.instance_id, dtype=torch.long),
            "image": torch.from_numpy(image_patch).unsqueeze(0).float(),
            "center_hint": torch.from_numpy(center_hint).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_patch).unsqueeze(0).float(),
            "gt_params": torch.from_numpy(gt_params).float(),
            "prompt_x": torch.tensor(meta.prompt_x, dtype=torch.float32),
            "prompt_y": torch.tensor(meta.prompt_y, dtype=torch.float32),
            "gt_x": torch.tensor(meta.gt_x, dtype=torch.float32),
            "gt_y": torch.tensor(meta.gt_y, dtype=torch.float32),
            "image_h": torch.tensor(meta.image_h, dtype=torch.long),
            "image_w": torch.tensor(meta.image_w, dtype=torch.long),
            "src_top": torch.tensor(meta.src_top, dtype=torch.long),
            "src_left": torch.tensor(meta.src_left, dtype=torch.long),
            "src_bottom": torch.tensor(meta.src_bottom, dtype=torch.long),
            "src_right": torch.tensor(meta.src_right, dtype=torch.long),
            "dst_top": torch.tensor(meta.dst_top, dtype=torch.long),
            "dst_left": torch.tensor(meta.dst_left, dtype=torch.long),
            "patch_size": torch.tensor(meta.patch_size, dtype=torch.long),
        }


def build_ellipse_dataset(config: DatasetConfig, split_name: str, image_dir_name: str, mask_dir_name: str, patch_size: int, prompt_noise_std: float, center_hint_sigma: float, seed: int = 42, cache_data: bool = False) -> CentroidConditionedEllipseDataset:
    return CentroidConditionedEllipseDataset(
        dataset_root=config.root,
        split_name=split_name,
        image_dir_name=image_dir_name,
        mask_dir_name=mask_dir_name,
        patch_size=patch_size,
        prompt_noise_std=prompt_noise_std,
        center_hint_sigma=center_hint_sigma,
        img_mean=config.img_mean,
        img_std=config.img_std,
        seed=seed,
        cache_data=cache_data,
    )


def build_ellipse_datasets(config: DatasetConfig, patch_size: int, train_prompt_noise_std: float, eval_prompt_noise_std: float, center_hint_sigma: float, seed: int = 42, cache_data: bool = False) -> tuple[CentroidConditionedEllipseDataset | None, CentroidConditionedEllipseDataset]:
    train_set = None
    if config.train_split not in (None, ""):
        train_set = build_ellipse_dataset(
            config=config,
            split_name=config.train_split,
            image_dir_name=config.train_image_dir,
            mask_dir_name=config.train_mask_dir,
            patch_size=patch_size,
            prompt_noise_std=train_prompt_noise_std,
            center_hint_sigma=center_hint_sigma,
            seed=seed,
            cache_data=cache_data,
        )
    val_set = build_ellipse_dataset(
        config=config,
        split_name=config.test_split,
        image_dir_name=config.test_image_dir,
        mask_dir_name=config.test_mask_dir,
        patch_size=patch_size,
        prompt_noise_std=eval_prompt_noise_std,
        center_hint_sigma=center_hint_sigma,
        seed=seed + 999,
        cache_data=cache_data,
    )
    return train_set, val_set


__all__ = ["CentroidConditionedEllipseDataset", "build_ellipse_dataset", "build_ellipse_datasets"]

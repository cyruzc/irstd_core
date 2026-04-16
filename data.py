"""
Data loading for image-mask segmentation datasets.

Two modes:
- FullSupervisionDataset: image + mask
- PointSupervisionDataset: image + mask + point label (centroid / coarse)

Train pipeline: normalize -> random crop -> augment (flip / transpose) -> to tensor
Test pipeline:  normalize -> pad to stride multiple -> to tensor (returns original size for inference crop)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from dataset_config import DATASET_REGISTRY, DatasetConfig, build_dataset_config, validate_dataset_config


# ============ Utility Functions ============

def load_split_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.float32)


def _resolve_optional(path_like: str | Path | None, base_dir: Path) -> Path | None:
    if path_like is None:
        return None
    p = Path(path_like)
    if p.is_absolute():
        return p
    candidate = p.resolve()
    return candidate if candidate.exists() else base_dir / p


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).unsqueeze(0).float()


# ============ Crop / Pad / Augment ============

def random_crop(
    image: np.ndarray,
    arrays: list[np.ndarray],
    patch_size: int,
    pos_prob: float = 0.5,
    rng: random.Random | None = None,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Random crop. With probability *pos_prob* crop around a positive pixel in *reference* (defaults to arrays[0])."""
    rng = rng or random
    h, w = image.shape
    need_pad = h < patch_size or w < patch_size
    if need_pad:
        pad_h = max(patch_size - h, 0)
        pad_w = max(patch_size - w, 0)
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        arrays = [np.pad(a, ((0, pad_h), (0, pad_w)), mode="constant") for a in arrays]
        h, w = image.shape

    ref = reference if reference is not None else arrays[0]
    ys, xs = np.where(ref > 0)
    if len(ys) > 0 and rng.random() < pos_prob:
        idx = rng.randrange(len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        top = min(max(cy - patch_size // 2, 0), h - patch_size)
        left = min(max(cx - patch_size // 2, 0), w - patch_size)
    else:
        top = rng.randint(0, h - patch_size)
        left = rng.randint(0, w - patch_size)

    out_arrays = [a[top:top + patch_size, left:left + patch_size] for a in arrays]
    return image[top:top + patch_size, left:left + patch_size], out_arrays


def pad_to_multiple(
    image: np.ndarray,
    arrays: list[np.ndarray],
    stride: int = 32,
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int]]:
    """Pad to nearest multiple of *stride*. Returns padded image, arrays, and original (h, w)."""
    h, w = image.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
        arrays = [np.pad(a, ((0, pad_h), (0, pad_w)), mode="constant") for a in arrays]
    return image, arrays, (h, w)


def augment(
    image: np.ndarray,
    arrays: list[np.ndarray],
    rng: random.Random,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Random flip + transpose augmentation. No-copy views, caller must not modify in-place."""
    all_arrays = [image, *arrays]

    if rng.random() < 0.5:  # vertical flip
        all_arrays = [a[::-1, :] for a in all_arrays]
    if rng.random() < 0.5:  # horizontal flip
        all_arrays = [a[:, ::-1] for a in all_arrays]
    if rng.random() < 0.5:  # transpose (equivalent to 90 deg rotation + flip)
        all_arrays = [a.T for a in all_arrays]

    # single copy to ensure contiguous memory for to_tensor
    all_arrays = [np.ascontiguousarray(a) for a in all_arrays]
    return all_arrays[0], all_arrays[1:]


# ============ Record Types ============

@dataclass(frozen=True)
class FullSampleRecord:
    name: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PointSampleRecord:
    name: str
    image_path: Path
    mask_path: Path
    point_label_path: Path


def resolve_full_records(
    dataset_root: Path,
    split_name: str,
    image_dir_name: str = "images",
    mask_dir_name: str = "masks",
) -> list[FullSampleRecord]:
    image_dir = dataset_root / image_dir_name
    mask_dir = dataset_root / mask_dir_name
    names = load_split_file(dataset_root / split_name)
    records: list[FullSampleRecord] = []
    for name in names:
        image_path = image_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        records.append(FullSampleRecord(name=name, image_path=image_path, mask_path=mask_path))
    return records


def resolve_point_records(
    dataset_root: Path,
    split_name: str,
    point_label_dir: str | Path,
    image_dir_name: str = "images",
    mask_dir_name: str = "masks",
) -> list[PointSampleRecord]:
    image_dir = dataset_root / image_dir_name
    mask_dir = dataset_root / mask_dir_name
    resolved_point_dir = _resolve_optional(point_label_dir, dataset_root)
    if resolved_point_dir is None:
        raise ValueError("point_label_dir is required for PointSupervisionDataset")

    names = load_split_file(dataset_root / split_name)
    records: list[PointSampleRecord] = []
    for name in names:
        image_path = image_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        point_path = resolved_point_dir / f"{name}.png"
        if not point_path.exists():
            raise FileNotFoundError(f"Missing point label: {point_path}")
        records.append(PointSampleRecord(
            name=name, image_path=image_path,
            mask_path=mask_path, point_label_path=point_path,
        ))
    return records


# ============ Full Supervision Dataset ============

class FullSupervisionDataset(Dataset):
    """
    image + mask dataset.

    Train: normalize -> random_crop -> augment -> to_tensor
    Test:  normalize -> pad_to_multiple -> to_tensor  (returns original_size)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split_name: str,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        train: bool = False,
        patch_size: int = 256,
        pos_prob: float = 0.5,
        pad_stride: int = 32,
        seed: int = 42,
        cache_data: bool = False,
        img_mean: float = 0.0,
        img_std: float = 1.0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.img_mean = img_mean
        self.img_std = img_std
        self.train = train
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.pad_stride = pad_stride
        self._seed = seed
        self.rng = random.Random(seed)
        self.records = resolve_full_records(
            self.dataset_root, split_name,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
        )
        self.cache_data = cache_data
        self.cached: list[tuple[np.ndarray, np.ndarray]] | None = None
        if self.cache_data:
            self.cached = [self._load(r) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load(record: FullSampleRecord) -> tuple[np.ndarray, np.ndarray]:
        image = load_grayscale(record.image_path)
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        record = self.records[index]
        image, mask = self.cached[index] if self.cached is not None else self._load(record)
        image = (image - self.img_mean) / self.img_std

        if self.train:
            image, (mask,) = random_crop(image, [mask], self.patch_size, self.pos_prob, self.rng)
            image, (mask,) = augment(image, [mask], self.rng)
            return {"name": record.name, "image": to_tensor(image), "mask": to_tensor(mask)}

        image, (mask,), original_size = pad_to_multiple(image, [mask], self.pad_stride)
        return {
            "name": record.name,
            "image": to_tensor(image),
            "mask": to_tensor(mask),
            "original_size": original_size,
        }


# ============ Point Supervision Dataset ============

class PointSupervisionDataset(Dataset):
    """
    image + mask + point label dataset.

    Train: normalize -> random_crop -> augment -> to_tensor
    Test:  normalize -> pad_to_multiple -> to_tensor  (returns original_size)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split_name: str,
        point_label_dir: str | Path,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        train: bool = False,
        patch_size: int = 256,
        pos_prob: float = 0.5,
        pad_stride: int = 32,
        seed: int = 42,
        cache_data: bool = False,
        img_mean: float = 0.0,
        img_std: float = 1.0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.img_mean = img_mean
        self.img_std = img_std
        self.train = train
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.pad_stride = pad_stride
        self._seed = seed
        self.rng = random.Random(seed)
        self.records = resolve_point_records(
            self.dataset_root, split_name,
            point_label_dir=point_label_dir,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
        )
        self.cache_data = cache_data
        self.cached: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        if self.cache_data:
            self.cached = [self._load(r) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load(record: PointSampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = load_grayscale(record.image_path)
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        point = (load_grayscale(record.point_label_path) > 0).astype(np.float32)
        return image, mask, point

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        record = self.records[index]
        image, mask, point = self.cached[index] if self.cached is not None else self._load(record)
        image = (image - self.img_mean) / self.img_std

        if self.train:
            ref = point if point.any() else mask
            image, (out_mask, out_point) = random_crop(
                image, [mask, point], self.patch_size, self.pos_prob, self.rng,
                reference=ref,
            )
            image, arrays = augment(image, [out_mask, out_point], self.rng)
            out_mask, out_point = arrays
            return {
                "name": record.name,
                "image": to_tensor(image),
                "mask": to_tensor(out_mask),
                "point": to_tensor(out_point),
            }

        image, (mask, point), original_size = pad_to_multiple(image, [mask, point], self.pad_stride)
        return {
            "name": record.name,
            "image": to_tensor(image),
            "mask": to_tensor(mask),
            "point": to_tensor(point),
            "original_size": original_size,
        }


def worker_init_fn(worker_id: int) -> None:
    """Re-seed per-worker RNG so augmentations are independent across workers."""
    info = get_worker_info()
    info.dataset.rng = random.Random(info.dataset._seed + worker_id)


__all__ = [
    "FullSupervisionDataset",
    "PointSupervisionDataset",
    "random_crop",
    "pad_to_multiple",
    "augment",
    "worker_init_fn",
    "load_grayscale",
    "load_split_file",
    "resolve_full_records",
    "resolve_point_records",
    "to_tensor",
]

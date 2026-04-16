from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import load_grayscale, load_split_file, resolve_full_records
from dataset_config import DatasetConfig
from point_model import make_point_heatmap


class PointDetectionDataset(Dataset):
    """Full-image dataset for point heatmap supervision."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_name: str,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        centroid_dir_name: str = "masks_centroid",
        sigma: float = 2.5,
        img_mean: float = 0.0,
        img_std: float = 1.0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sigma = sigma
        self.img_mean = img_mean
        self.img_std = img_std
        self.centroid_dir = self.dataset_root / centroid_dir_name

        self.records = resolve_full_records(
            self.dataset_root, split_name,
            image_dir_name=image_dir_name, mask_dir_name=mask_dir_name,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = load_grayscale(record.image_path)
        image = ((image - self.img_mean) / self.img_std).astype(np.float32)

        # Extract centroid points from centroid label or mask
        points = self._extract_points(record.name, record.mask_path)

        # Generate heatmap
        heatmap = make_point_heatmap(image.shape[0], image.shape[1], points, sigma=self.sigma)

        return {
            "name": record.name,
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0).float(),
            "num_points": torch.tensor(len(points), dtype=torch.long),
        }

    def _extract_points(self, name: str, mask_path: Path) -> list[tuple[float, float]]:
        """Extract point coordinates from centroid label image."""
        centroid_path = self.centroid_dir / f"{name}.png"
        if centroid_path.exists():
            centroid_img = load_grayscale(centroid_path)
            ys, xs = np.where(centroid_img > 128)
            return [(float(xs[i]), float(ys[i])) for i in range(len(xs))]
        # Fallback: extract centroids from mask
        from ellipse_utils import component_instances
        mask = (load_grayscale(mask_path) > 0).astype(np.float32)
        return [(float(inst["centroid_x"]), float(inst["centroid_y"])) for inst in component_instances(mask)]


def build_point_datasets(
    config: DatasetConfig,
    sigma: float = 2.5,
) -> tuple[PointDetectionDataset | None, PointDetectionDataset]:
    train_set = None
    if config.train_split not in (None, ""):
        train_set = PointDetectionDataset(
            dataset_root=config.root, split_name=config.train_split,
            image_dir_name=config.train_image_dir, mask_dir_name=config.train_mask_dir,
            centroid_dir_name=config.centroid_label_dir or "masks_centroid",
            sigma=sigma, img_mean=config.img_mean, img_std=config.img_std,
        )
    val_set = PointDetectionDataset(
        dataset_root=config.root, split_name=config.test_split,
        image_dir_name=config.test_image_dir, mask_dir_name=config.test_mask_dir,
        centroid_dir_name=config.centroid_label_dir or "masks_centroid",
        sigma=sigma, img_mean=config.img_mean, img_std=config.img_std,
    )
    return train_set, val_set


__all__ = ["PointDetectionDataset", "build_point_datasets"]

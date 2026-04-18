from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for image-mask segmentation datasets."""

    name: str
    root: Path
    train_image_dir: str
    train_mask_dir: str
    test_image_dir: str
    test_mask_dir: str
    train_split: str | None
    test_split: str
    centroid_label_dir: str | None = None
    coarse_label_dir: str | None = None
    img_mean: float = 0.0
    img_std: float = 1.0


_DATA_ROOT = Path(__file__).parent / "datasets"

DATASET_REGISTRY = {
    "sirst3": DatasetConfig(
        name="sirst3",
        root=_DATA_ROOT / "SIRST3",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_SIRST3.txt",
        test_split="img_idx/test_SIRST3.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=95.010,
        img_std=41.511,
    ),
    "irstd1k": DatasetConfig(
        name="irstd1k",
        root=_DATA_ROOT / "IRSTD-1K",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_IRSTD-1K.txt",
        test_split="img_idx/test_IRSTD-1K.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=87.466,
        img_std=39.720,
    ),
    "nuaa_sirst": DatasetConfig(
        name="nuaa_sirst",
        root=_DATA_ROOT / "NUAA-SIRST",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_NUAA-SIRST.txt",
        test_split="img_idx/test_NUAA-SIRST.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=101.064,
        img_std=34.620,
    ),
    "nudt_sirst": DatasetConfig(
        name="nudt_sirst",
        root=_DATA_ROOT / "NUDT-SIRST",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_NUDT-SIRST.txt",
        test_split="img_idx/test_NUDT-SIRST.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=95.0,
        img_std=40.0,
    ),
}


def build_dataset_config(args) -> DatasetConfig:
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name:
        try:
            config = DATASET_REGISTRY[dataset_name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {dataset_name}") from exc
    else:
        dataset_root = getattr(args, "dataset_root", None)
        if dataset_root in (None, ""):
            raise ValueError("Either --dataset-name or --dataset-root must be provided.")
        config = DatasetConfig(
            name="custom",
            root=Path(dataset_root),
            train_image_dir=getattr(args, "image_dir_name", None) or "images",
            train_mask_dir=getattr(args, "mask_dir_name", None) or "masks",
            test_image_dir=getattr(args, "test_image_dir_name", None) or getattr(args, "image_dir_name", None) or "images",
            test_mask_dir=getattr(args, "test_mask_dir_name", None) or getattr(args, "mask_dir_name", None) or "masks",
            train_split=getattr(args, "train_split", None),
            test_split=getattr(args, "test_split", None) or "test.txt",
        )

    return replace(
        config,
        root=Path(getattr(args, "dataset_root", None) or str(config.root)),
        train_image_dir=getattr(args, "image_dir_name", None) or config.train_image_dir,
        train_mask_dir=getattr(args, "mask_dir_name", None) or config.train_mask_dir,
        test_image_dir=getattr(args, "test_image_dir_name", None) or config.test_image_dir,
        test_mask_dir=getattr(args, "test_mask_dir_name", None) or config.test_mask_dir,
        train_split=getattr(args, "train_split", None) or config.train_split,
        test_split=getattr(args, "test_split", None) or config.test_split,
    )


def validate_dataset_config(config: DatasetConfig, require_train_split: bool = True) -> None:
    if not config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.root}")
    if require_train_split and config.train_split in (None, ""):
        raise ValueError("Training requires a dataset with a train split.")
    if config.test_split in (None, ""):
        raise ValueError("Dataset config is missing a test split.")


__all__ = [
    "DatasetConfig",
    "DATASET_REGISTRY",
    "build_dataset_config",
    "validate_dataset_config",
]

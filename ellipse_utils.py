from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import measure


EPS = 1e-6


@dataclass(frozen=True)
class CropMeta:
    src_top: int
    src_left: int
    src_bottom: int
    src_right: int
    dst_top: int
    dst_left: int
    patch_size: int
    image_h: int
    image_w: int


@dataclass(frozen=True)
class EllipseFit:
    cx: float
    cy: float
    a: float
    b: float
    phi: float
    area: float


@dataclass(frozen=True)
class InstanceRecord:
    name: str
    image_path: Path
    mask_path: Path
    instance_id: int
    centroid_x: float
    centroid_y: float
    area: int
    bbox: tuple[int, int, int, int]


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


def component_instances(mask: np.ndarray) -> list[dict]:
    binary = ensure_binary(mask)
    labeled = measure.label(binary, connectivity=2)
    instances: list[dict] = []
    for idx, region in enumerate(measure.regionprops(labeled), start=1):
        comp = (labeled == region.label).astype(np.float32)
        minr, minc, maxr, maxc = region.bbox
        cy, cx = region.centroid
        instances.append(
            {
                "instance_id": idx - 1,
                "mask": comp,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "area": int(region.area),
                "bbox": (int(minr), int(minc), int(maxr), int(maxc)),
            }
        )
    return instances


def fit_ellipse_from_mask(mask: np.ndarray) -> EllipseFit:
    binary = ensure_binary(mask)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise ValueError("Cannot fit ellipse to an empty mask.")

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    cx = float(x.mean())
    cy = float(y.mean())

    x_centered = x - cx
    y_centered = y - cy
    coords = np.stack([x_centered, y_centered], axis=0)
    cov = np.cov(coords, bias=True)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, EPS)
    a = 2.0 * float(np.sqrt(eigvals[0]))
    b = 2.0 * float(np.sqrt(eigvals[1]))

    area = float(binary.sum())
    raw_area = np.pi * a * b
    if raw_area > EPS:
        scale = float(np.sqrt(area / raw_area))
        a *= scale
        b *= scale

    direction = eigvecs[:, 0]
    phi = float(np.arctan2(direction[1], direction[0]))
    phi = normalize_half_pi(phi)
    if b > a:
        a, b = b, a
        phi = normalize_half_pi(phi + np.pi / 2.0)

    return EllipseFit(cx=cx, cy=cy, a=max(a, 0.5), b=max(b, 0.5), phi=phi, area=area)


def normalize_half_pi(phi: float) -> float:
    return float(((phi + np.pi / 2.0) % np.pi) - np.pi / 2.0)


def crop_with_pad(array: np.ndarray, center_x: float, center_y: float, patch_size: int, pad_value: float = 0.0) -> tuple[np.ndarray, CropMeta]:
    h, w = array.shape
    half = patch_size // 2
    top = int(round(center_y)) - half
    left = int(round(center_x)) - half
    bottom = top + patch_size
    right = left + patch_size

    src_top = max(top, 0)
    src_left = max(left, 0)
    src_bottom = min(bottom, h)
    src_right = min(right, w)

    dst_top = src_top - top
    dst_left = src_left - left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    patch = np.full((patch_size, patch_size), pad_value, dtype=array.dtype)
    patch[dst_top:dst_bottom, dst_left:dst_right] = array[src_top:src_bottom, src_left:src_right]

    meta = CropMeta(
        src_top=src_top,
        src_left=src_left,
        src_bottom=src_bottom,
        src_right=src_right,
        dst_top=dst_top,
        dst_left=dst_left,
        patch_size=patch_size,
        image_h=h,
        image_w=w,
    )
    return patch, meta


def paste_patch(canvas: np.ndarray, patch: np.ndarray, meta: CropMeta, reduce: str = "max") -> np.ndarray:
    src_h = meta.src_bottom - meta.src_top
    src_w = meta.src_right - meta.src_left
    dst_bottom = meta.dst_top + src_h
    dst_right = meta.dst_left + src_w
    patch_valid = patch[meta.dst_top:dst_bottom, meta.dst_left:dst_right]

    if reduce == "max":
        canvas[meta.src_top:meta.src_bottom, meta.src_left:meta.src_right] = np.maximum(
            canvas[meta.src_top:meta.src_bottom, meta.src_left:meta.src_right],
            patch_valid,
        )
    elif reduce == "overwrite":
        canvas[meta.src_top:meta.src_bottom, meta.src_left:meta.src_right] = patch_valid
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")
    return canvas


def make_center_hint(patch_size: int, sigma: float = 2.0, dtype: np.dtype = np.float32) -> np.ndarray:
    center = (patch_size - 1) / 2.0
    ys, xs = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="ij")
    dist2 = (xs - center) ** 2 + (ys - center) ** 2
    hint = np.exp(-0.5 * dist2 / max(sigma ** 2, EPS))
    return hint.astype(dtype)


def rasterize_ellipse_numpy(height: int, width: int, cx: float, cy: float, a: float, b: float, phi: float, threshold: float = 1.0) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    x = xs - np.float32(cx)
    y = ys - np.float32(cy)
    cos_phi = np.cos(np.float32(phi))
    sin_phi = np.sin(np.float32(phi))
    x_rot = x * cos_phi + y * sin_phi
    y_rot = -x * sin_phi + y * cos_phi
    value = (x_rot / max(a, 0.5)) ** 2 + (y_rot / max(b, 0.5)) ** 2
    return (value <= threshold).astype(np.float32)


def sample_prompt_center(gt_x: float, gt_y: float, noise_std: float, rng: np.random.Generator) -> tuple[float, float]:
    if noise_std <= 0:
        return gt_x, gt_y
    dx, dy = rng.normal(loc=0.0, scale=noise_std, size=2)
    return float(gt_x + dx), float(gt_y + dy)


def angle_abs_error(pred_phi: float, gt_phi: float) -> float:
    diff = abs(pred_phi - gt_phi)
    diff = min(diff, abs(diff - np.pi), abs(diff + np.pi))
    if diff > np.pi / 2.0:
        diff = np.pi - diff
    return float(diff)


__all__ = [
    "CropMeta",
    "EllipseFit",
    "InstanceRecord",
    "component_instances",
    "fit_ellipse_from_mask",
    "normalize_half_pi",
    "crop_with_pad",
    "paste_patch",
    "make_center_hint",
    "rasterize_ellipse_numpy",
    "sample_prompt_center",
    "angle_abs_error",
]

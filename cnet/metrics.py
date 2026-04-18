"""
Evaluation metrics for binary segmentation.

Accumulator-style (update/get/reset):
- IoUMetric:  IoU (dataset-level aggregate)
- nIoUMetric: nIoU (per-sample IoU averaged)
- PD_FA_Metric: PD (probability of detection) and FA (false alarm rate per pixel)

All update() calls accept torch tensors [B, 1, H, W] with values in [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch
from skimage import measure


def _to_numpy_binary(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    return (tensor.detach().cpu().numpy() > threshold).astype(np.int64)


# ============ Fast GPU IoU/nIoU (for per-epoch eval) ============

class FastIoU:
    """Pure-torch IoU + nIoU. Stays on GPU, no numpy/skimage overhead.

    Usage (per epoch):
        meter = FastIoU()
        for batch in loader:
            meter.update(pred, mask)
        print(meter.get())   # {"IoU": ..., "nIoU": ...}
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.total_inter = torch.tensor(0.0)
        self.total_union = torch.tensor(0.0)
        self.sample_ious: list[float] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        p = (pred > self.threshold).float()
        t = (target > self.threshold).float()
        tp = p * t
        # ---- dataset-level (aggregate) ----
        self.total_inter = self.total_inter.to(tp.device) + tp.sum()
        self.total_union = self.total_union.to(tp.device) + p.sum() + t.sum() - tp.sum()
        # ---- per-sample (vectorized) ----
        b = pred.shape[0]
        flat_dim = int(p[0].numel())
        p_flat = p.reshape(b, flat_dim)
        t_flat = t.reshape(b, flat_dim)
        tp_flat = p_flat * t_flat
        inters = tp_flat.sum(dim=1)
        unions = p_flat.sum(dim=1) + t_flat.sum(dim=1) - inters
        sample_iou = (inters / (unions + 1e-10)).cpu().tolist()
        self.sample_ious.extend(sample_iou)

    def get(self) -> dict[str, float]:
        iou = (self.total_inter / (self.total_union + 1e-10)).item()
        niou = float(np.mean(self.sample_ious)) if self.sample_ious else 0.0
        return {"IoU": iou, "nIoU": niou}


# ============ PD / FA ============

class PD_FA_Metric:
    """
    PD:  probability of detection = matched_targets / total_targets
    FA:  false alarm rate = false_alarm_pixels / total_pixels

    A predicted connected component is "matched" if its centroid is within
    *distance_thresh* pixels of a ground-truth centroid.
    """

    def __init__(self, distance_thresh: float = 3.0, threshold: float = 0.5) -> None:
        self.distance_thresh = distance_thresh
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.matched = 0
        self.total_targets = 0
        self.false_alarm_pixels = 0
        self.total_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = _to_numpy_binary(pred, self.threshold)
        target_np = _to_numpy_binary(target, self.threshold)

        for b in range(pred_np.shape[0]):
            p = pred_np[b, 0] if pred_np.ndim == 4 else pred_np[b]
            t = target_np[b, 0] if target_np.ndim == 4 else target_np[b]

            self.total_pixels += p.size

            pred_regions = measure.regionprops(measure.label(p, connectivity=2))
            gt_regions = measure.regionprops(measure.label(t, connectivity=2))

            self.total_targets += len(gt_regions)

            matched_pred_indices: set[int] = set()
            for gt in gt_regions:
                gt_centroid = np.array(gt.centroid)
                for m, pr in enumerate(pred_regions):
                    if m in matched_pred_indices:
                        continue
                    dist = np.linalg.norm(np.array(pr.centroid) - gt_centroid)
                    if dist < self.distance_thresh:
                        matched_pred_indices.add(m)
                        self.matched += 1
                        break

            matched_pixels = sum(
                pred_regions[m].area for m in matched_pred_indices
            )
            self.false_alarm_pixels += int(p.sum()) - matched_pixels

    def get(self) -> tuple[float, float]:
        pd = self.matched / (self.total_targets + 1e-10)
        fa = self.false_alarm_pixels / (self.total_pixels + 1e-10)
        return pd, fa


# ============ Convenience ============

class SegMetrics:
    """All metrics in one accumulator."""

    def __init__(
        self,
        threshold: float = 0.5,
        distance_thresh: float = 3.0,
    ) -> None:
        self.fast_iou = FastIoU(threshold)
        self.pd_fa = PD_FA_Metric(distance_thresh, threshold)

    def reset(self) -> None:
        self.fast_iou.reset()
        self.pd_fa.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.fast_iou.update(pred, target)
        self.pd_fa.update(pred, target)

    def get(self) -> dict[str, float]:
        pd, fa = self.pd_fa.get()
        iou_results = self.fast_iou.get()
        return {
            "IoU": iou_results["IoU"],
            "nIoU": iou_results["nIoU"],
            "PD": pd,
            "FA": fa,
        }


__all__ = [
    "FastIoU",
    "PD_FA_Metric",
    "SegMetrics",
]

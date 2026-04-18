from __future__ import annotations

import torch
import torch.nn.functional as F


def render_gaussian_targets(point_map: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    if sigma <= 0:
        return point_map.float()

    radius = max(int(3 * sigma), 1)
    coords = torch.arange(-radius, radius + 1, device=point_map.device, dtype=point_map.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.max().clamp_min(1e-6)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    target = F.conv2d(point_map.float(), kernel, padding=radius)
    return target.clamp_(0.0, 1.0)


def render_ring_targets(
    point_map: torch.Tensor,
    inner_radius: int = 2,
    outer_radius: int = 6,
) -> torch.Tensor:
    if outer_radius <= inner_radius:
        return torch.zeros_like(point_map)
    kernel_outer = 2 * outer_radius + 1
    kernel_inner = 2 * inner_radius + 1
    outer = F.max_pool2d(point_map.float(), kernel_size=kernel_outer, stride=1, padding=outer_radius)
    inner = F.max_pool2d(point_map.float(), kernel_size=kernel_inner, stride=1, padding=inner_radius)
    ring = (outer > 0).float() * (1.0 - (inner > 0).float())
    return ring


def build_offset_targets(
    point_map: torch.Tensor,
    support_radius: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = point_map.shape
    offsets = point_map.new_zeros((b, 2, h, w))
    mask = point_map.new_zeros((b, 1, h, w))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=point_map.device, dtype=point_map.dtype),
        torch.arange(w, device=point_map.device, dtype=point_map.dtype),
        indexing="ij",
    )

    for i in range(b):
        points = (point_map[i, 0] > 0.5).nonzero(as_tuple=False).float()
        if points.numel() == 0:
            continue

        py = points[:, 0][:, None, None]
        px = points[:, 1][:, None, None]
        dist2 = (yy.unsqueeze(0) - py).square() + (xx.unsqueeze(0) - px).square()
        min_dist2, nearest = dist2.min(dim=0)
        support = min_dist2 <= float(support_radius * support_radius)
        if not support.any():
            continue

        nearest_points = points[nearest]
        dy = nearest_points[..., 0] - yy
        dx = nearest_points[..., 1] - xx
        scale = float(max(support_radius, 1))
        offsets[i, 0] = dx / scale
        offsets[i, 1] = dy / scale
        mask[i, 0] = support.float()

    return offsets, mask


def accumulate_votes(
    score_map: torch.Tensor,
    offset_map: torch.Tensor,
    vote_radius: float = 5.0,
) -> torch.Tensor:
    if score_map.ndim != 4 or score_map.shape[1] != 1:
        raise ValueError(f"Expected [B, 1, H, W] score map, got {tuple(score_map.shape)}")
    if offset_map.shape[:2] != (score_map.shape[0], 2):
        raise ValueError(f"Expected offset map [B, 2, H, W], got {tuple(offset_map.shape)}")

    b, _, h, w = score_map.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=score_map.device, dtype=score_map.dtype),
        torch.arange(w, device=score_map.device, dtype=score_map.dtype),
        indexing="ij",
    )
    yy = yy.unsqueeze(0).expand(b, -1, -1)
    xx = xx.unsqueeze(0).expand(b, -1, -1)

    vote_x = (xx + offset_map[:, 0] * vote_radius).round().long().clamp_(0, w - 1)
    vote_y = (yy + offset_map[:, 1] * vote_radius).round().long().clamp_(0, h - 1)

    flat_indices = vote_y * w + vote_x
    flat_votes = score_map[:, 0].reshape(b, -1)
    flat_indices = flat_indices.reshape(b, -1)

    vote_map = score_map.new_zeros((b, h * w))
    vote_map.scatter_add_(1, flat_indices, flat_votes)
    vote_map = vote_map.view(b, 1, h, w)
    vote_map = vote_map / vote_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return vote_map


def extract_candidate_centers(
    score_map: torch.Tensor,
    topk: int = 16,
    nms_kernel: int = 7,
    score_threshold: float = 0.3,
) -> dict[str, torch.Tensor]:
    if score_map.ndim != 4 or score_map.shape[1] != 1:
        raise ValueError(f"Expected [B, 1, H, W] score map, got {tuple(score_map.shape)}")
    if nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be odd.")

    pooled = F.max_pool2d(score_map, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
    peaks = score_map * (score_map == pooled).to(score_map.dtype)
    b, _, h, w = peaks.shape
    flat = peaks.flatten(2).squeeze(1)
    scores, indices = flat.topk(k=min(topk, h * w), dim=-1)
    valid = scores >= score_threshold
    ys = torch.div(indices, w, rounding_mode="floor")
    xs = indices % w
    coords = torch.stack([ys, xs], dim=-1)
    return {"scores": scores, "coords": coords, "valid_mask": valid}


def pairwise_redundancy_loss(candidates: dict[str, torch.Tensor], radius: float = 6.0) -> torch.Tensor:
    scores = candidates["scores"]
    coords = candidates["coords"].float()
    valid = candidates["valid_mask"]
    if scores.numel() == 0:
        return scores.new_tensor(0.0)

    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist2 = diff.square().sum(dim=-1)
    proximity = torch.exp(-dist2 / max(2.0 * radius * radius, 1e-6))
    score_pairs = scores.unsqueeze(2) * scores.unsqueeze(1)
    valid_pairs = (valid.unsqueeze(2) & valid.unsqueeze(1)).to(scores.dtype)
    eye = torch.eye(scores.shape[1], device=scores.device, dtype=scores.dtype).unsqueeze(0)
    pair_mask = valid_pairs * (1.0 - eye)
    denom = pair_mask.sum().clamp_min(1.0)
    return (score_pairs * proximity * pair_mask).sum() / denom


@torch.no_grad()
def match_candidates(
    candidate_coords: torch.Tensor,
    candidate_valid: torch.Tensor,
    gt_points: torch.Tensor,
    distance_thresh: float = 3.0,
) -> tuple[int, int, int]:
    total_matched = 0
    total_pred = 0
    total_gt = 0
    gt_binary = gt_points > 0.5

    for b in range(gt_points.shape[0]):
        gt_coords = gt_binary[b, 0].nonzero(as_tuple=False).float()
        pred_coords = candidate_coords[b][candidate_valid[b]].float()
        total_gt += int(gt_coords.shape[0])
        total_pred += int(pred_coords.shape[0])
        if gt_coords.numel() == 0 or pred_coords.numel() == 0:
            continue

        distances = torch.cdist(pred_coords, gt_coords)
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        pairs = torch.nonzero(distances <= distance_thresh, as_tuple=False)
        if pairs.numel() == 0:
            continue
        ordered = torch.argsort(distances[pairs[:, 0], pairs[:, 1]])
        for idx in ordered.tolist():
            pred_idx = int(pairs[idx, 0])
            gt_idx = int(pairs[idx, 1])
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
        total_matched += len(matched_gt)

    return total_matched, total_pred, total_gt

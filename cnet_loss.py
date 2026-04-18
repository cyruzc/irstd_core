from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnet_utils import (
    build_offset_targets,
    extract_candidate_centers,
    pairwise_redundancy_loss,
    render_gaussian_targets,
    render_ring_targets,
)


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((numerator + 1e-6) / (denominator + 1e-6)).mean()


def weighted_point_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos_mask = target > 0
    pos_count = pos_mask.sum().clamp_min(1).to(logits.dtype)
    neg_count = (~pos_mask).sum().clamp_min(1).to(logits.dtype)
    pos_weight = (neg_count / pos_count).clamp(max=200.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def peak_ranking_loss(
    proposal_map: torch.Tensor,
    point_map: torch.Tensor,
    inner_radius: int,
    outer_radius: int,
    margin: float,
) -> torch.Tensor:
    """Force each GT neighborhood to have a dominant local peak over its surrounding ring."""
    if outer_radius <= inner_radius:
        return proposal_map.sum() * 0.0

    positive_exclusion = F.max_pool2d(
        point_map.float(),
        kernel_size=2 * inner_radius + 1,
        stride=1,
        padding=inner_radius,
    ) > 0

    losses = []
    _, _, h, w = point_map.shape
    for b in range(point_map.shape[0]):
        gt_points = (point_map[b, 0] > 0.5).nonzero(as_tuple=False)
        if gt_points.numel() == 0:
            continue

        sample_map = proposal_map[b, 0]
        exclusion_map = positive_exclusion[b, 0]
        for yx in gt_points.tolist():
            cy, cx = int(yx[0]), int(yx[1])
            top = max(cy - outer_radius, 0)
            bottom = min(cy + outer_radius + 1, h)
            left = max(cx - outer_radius, 0)
            right = min(cx + outer_radius + 1, w)

            patch = sample_map[top:bottom, left:right]
            patch_h, patch_w = patch.shape
            yy, xx = torch.meshgrid(
                torch.arange(top, bottom, device=patch.device),
                torch.arange(left, right, device=patch.device),
                indexing="ij",
            )
            dist2 = (yy - cy).square() + (xx - cx).square()
            inner_mask = dist2 <= inner_radius * inner_radius
            ring_mask = (dist2 <= outer_radius * outer_radius) & (~inner_mask)
            ring_mask = ring_mask & (~exclusion_map[top:bottom, left:right])

            pos_scores = patch[inner_mask]
            if pos_scores.numel() == 0:
                continue
            pos_peak = pos_scores.max()

            neg_scores = patch[ring_mask]
            if neg_scores.numel() == 0:
                neg_peak = patch.new_tensor(0.0)
            else:
                neg_peak = neg_scores.max()

            losses.append(F.relu(margin - pos_peak + neg_peak))

    if not losses:
        return proposal_map.sum() * 0.0
    return torch.stack(losses).mean()


class CandidateFormationLoss(nn.Module):
    """Center-voting candidate learning with ring suppression and offset supervision."""

    def __init__(
        self,
        sigma: float = 2.0,
        ring_inner_radius: int = 2,
        ring_outer_radius: int = 6,
        vote_radius: int = 5,
        target_weight: float = 1.0,
        dice_weight: float = 0.5,
        vote_weight: float = 1.0,
        ring_weight: float = 0.5,
        offset_weight: float = 1.0,
        peak_ranking_weight: float = 0.2,
        peak_ranking_margin: float = 0.15,
        count_weight: float = 0.05,
        over_count_weight: float = 1.0,
        under_count_weight: float = 0.25,
        redundancy_weight: float = 0.2,
        redundancy_radius: float = 6.0,
        topk: int = 16,
        nms_kernel: int = 7,
        score_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.ring_inner_radius = ring_inner_radius
        self.ring_outer_radius = ring_outer_radius
        self.vote_radius = vote_radius
        self.target_weight = target_weight
        self.dice_weight = dice_weight
        self.vote_weight = vote_weight
        self.ring_weight = ring_weight
        self.offset_weight = offset_weight
        self.peak_ranking_weight = peak_ranking_weight
        self.peak_ranking_margin = peak_ranking_margin
        self.count_weight = count_weight
        self.over_count_weight = over_count_weight
        self.under_count_weight = under_count_weight
        self.redundancy_weight = redundancy_weight
        self.redundancy_radius = redundancy_radius
        self.topk = topk
        self.nms_kernel = nms_kernel
        self.score_threshold = score_threshold
        self.last_components: dict[str, float] = {}

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        point = batch["point"].float()
        center_logits = outputs["center_logits"]
        center_prob = outputs["center_prob"]
        vote_map = outputs["vote_map"]
        proposal_map = outputs["proposal_map"]
        offset_map = outputs["offset_map"]

        gaussian_target = render_gaussian_targets(point, sigma=self.sigma)
        ring_target = render_ring_targets(
            point,
            inner_radius=self.ring_inner_radius,
            outer_radius=self.ring_outer_radius,
        )
        offset_target, offset_mask = build_offset_targets(point, support_radius=self.vote_radius)

        center_loss = weighted_point_bce(center_logits, gaussian_target)
        dice_loss = soft_dice_loss(center_prob, gaussian_target)
        vote_loss = soft_dice_loss(proposal_map, gaussian_target) + weighted_point_bce(torch.logit(proposal_map.clamp(1e-4, 1 - 1e-4)), gaussian_target)
        ring_loss = (center_prob * ring_target).sum(dim=(1, 2, 3)) / ring_target.sum(dim=(1, 2, 3)).clamp_min(1.0)
        ring_loss = ring_loss.mean()
        ranking_loss = peak_ranking_loss(
            proposal_map,
            point,
            inner_radius=self.ring_inner_radius,
            outer_radius=self.ring_outer_radius,
            margin=self.peak_ranking_margin,
        )

        if offset_mask.any():
            offset_loss = F.smooth_l1_loss(
                offset_map * offset_mask,
                offset_target * offset_mask,
                reduction="sum",
            ) / offset_mask.sum().clamp_min(1.0)
        else:
            offset_loss = offset_map.sum() * 0.0

        candidates = {
            "scores": outputs.get("scores"),
            "coords": outputs.get("coords"),
            "valid_mask": outputs.get("valid_mask"),
        }
        if candidates["scores"] is None:
            candidates = extract_candidate_centers(
                proposal_map,
                topk=self.topk,
                nms_kernel=self.nms_kernel,
                score_threshold=self.score_threshold,
            )

        # Use a soft count proxy so this term stays differentiable.
        # A confident unique peak should contribute close to 1, while weak or redundant peaks stay near 0.
        pred_count = candidates["scores"].sum(dim=1)
        gt_count = point.flatten(1).sum(dim=1)
        over_count = torch.relu(pred_count - gt_count)
        under_count = torch.relu(gt_count - pred_count)
        count_loss = (
            self.over_count_weight * over_count.square()
            + self.under_count_weight * under_count.square()
        ).mean()
        redundancy_loss = pairwise_redundancy_loss(candidates, radius=self.redundancy_radius)

        loss = (
            self.target_weight * center_loss
            + self.dice_weight * dice_loss
            + self.vote_weight * vote_loss
            + self.ring_weight * ring_loss
            + self.offset_weight * offset_loss
            + self.peak_ranking_weight * ranking_loss
            + self.count_weight * count_loss
            + self.redundancy_weight * redundancy_loss
        )
        self.last_components = {
            "loss_total": float(loss.detach().item()),
            "loss_center": float(center_loss.detach().item()),
            "loss_dice": float(dice_loss.detach().item()),
            "loss_vote": float(vote_loss.detach().item()),
            "loss_ring": float(ring_loss.detach().item()),
            "loss_offset": float(offset_loss.detach().item()),
            "loss_peak_rank": float(ranking_loss.detach().item()),
            "loss_count": float(count_loss.detach().item()),
            "loss_redundancy": float(redundancy_loss.detach().item()),
        }
        return loss

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cde_losses import soft_dice_loss, soft_iou_loss, safe_binary_cross_entropy, mask_moments, moment_loss

EPS = 1e-6


def base_parameter_loss(decoded: dict[str, torch.Tensor], gt_base_params: torch.Tensor) -> torch.Tensor:
    pred = torch.stack(
        [decoded["dx"], decoded["dy"], decoded["a"], decoded["b"], decoded["phi"]],
        dim=1,
    )
    reg = F.smooth_l1_loss(pred[:, :4], gt_base_params[:, :4])
    angle = 1.0 - torch.cos(2.0 * (pred[:, 4] - gt_base_params[:, 4]))
    return reg + angle.mean()


def profile_loss(
    pred_profile: torch.Tensor,
    gt_profile: torch.Tensor,
) -> torch.Tensor:
    """Direct L1 supervision on radial profile in canonical coords."""
    return F.smooth_l1_loss(pred_profile, gt_profile)


class RadialProfileEllipseLoss(nn.Module):
    def __init__(
        self,
        w_dice: float = 1.0,
        w_iou: float = 1.0,
        w_bce: float = 0.5,
        w_param: float = 0.25,
        w_profile: float = 1.0,
        w_moment: float = 0.0,
        moment_warmup_epochs: int = 0,
        moment_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_iou = w_iou
        self.w_bce = w_bce
        self.w_param = w_param
        self.w_profile = w_profile
        self.w_moment = w_moment
        self.moment_warmup_epochs = int(moment_warmup_epochs)
        self.moment_weights = moment_weights

    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        decoded: dict[str, torch.Tensor],
        gt_base_params: torch.Tensor,
        gt_profile: torch.Tensor | None = None,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_dice = soft_dice_loss(pred_mask, target_mask)
        loss_iou = soft_iou_loss(pred_mask, target_mask)
        loss_bce = safe_binary_cross_entropy(pred_mask, target_mask)
        loss_param = base_parameter_loss(decoded, gt_base_params)

        if gt_profile is not None:
            loss_prof = profile_loss(decoded["profile"], gt_profile)
        else:
            loss_prof = pred_mask.new_tensor(0.0)

        if current_epoch >= self.moment_warmup_epochs and self.w_moment > 0:
            loss_moment = moment_loss(pred_mask, target_mask, weights=self.moment_weights)
        else:
            loss_moment = pred_mask.new_tensor(0.0)

        total = (
            self.w_dice * loss_dice
            + self.w_iou * loss_iou
            + self.w_bce * loss_bce
            + self.w_param * loss_param
            + self.w_profile * loss_prof
            + self.w_moment * loss_moment
        )
        stats = {
            "loss_total": float(total.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_iou": float(loss_iou.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_param": float(loss_param.detach().item()),
            "loss_profile": float(loss_prof.detach().item()),
            "loss_moment": float(loss_moment.detach().item()),
        }
        return total, stats


__all__ = [
    "RadialProfileEllipseLoss",
    "base_parameter_loss",
    "profile_loss",
]

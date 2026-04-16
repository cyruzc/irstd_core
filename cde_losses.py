from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + EPS) / (denom + EPS)
    return 1.0 - dice.mean()


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter + EPS) / (union + EPS)
    return 1.0 - iou.mean()


def safe_binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    device_type = pred.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pred_fp32 = pred.float().clamp(EPS, 1.0 - EPS)
        target_fp32 = target.float()
        return F.binary_cross_entropy(pred_fp32, target_fp32)


DEFAULT_MOMENT_WEIGHTS: dict[str, float] = {
    "area": 1.0,
    "cx": 2.0,
    "cy": 2.0,
    "mu20": 1.0,
    "mu02": 1.0,
    "mu11": 1.0,
}


def mask_moments(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W].")
    _, _, h, w = mask.shape
    device = mask.device
    dtype = mask.dtype

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, h, w)
    ys = ys.view(1, 1, h, w)

    mass = mask.sum(dim=(1, 2, 3), keepdim=True) + EPS
    cx = (mask * xs).sum(dim=(1, 2, 3), keepdim=True) / mass
    cy = (mask * ys).sum(dim=(1, 2, 3), keepdim=True) / mass
    x_centered = xs - cx
    y_centered = ys - cy
    mu20 = (mask * x_centered.pow(2)).sum(dim=(1, 2, 3), keepdim=True) / mass
    mu02 = (mask * y_centered.pow(2)).sum(dim=(1, 2, 3), keepdim=True) / mass
    mu11 = (mask * x_centered * y_centered).sum(dim=(1, 2, 3), keepdim=True) / mass
    return {
        "area": mass.squeeze(-1).squeeze(-1).squeeze(-1),
        "cx": cx.squeeze(-1).squeeze(-1).squeeze(-1),
        "cy": cy.squeeze(-1).squeeze(-1).squeeze(-1),
        "mu20": mu20.squeeze(-1).squeeze(-1).squeeze(-1),
        "mu02": mu02.squeeze(-1).squeeze(-1).squeeze(-1),
        "mu11": mu11.squeeze(-1).squeeze(-1).squeeze(-1),
    }


def moment_loss(pred: torch.Tensor, target: torch.Tensor, weights: dict[str, float] | None = None) -> torch.Tensor:
    pm = mask_moments(pred)
    tm = mask_moments(target)
    w = weights if weights is not None else DEFAULT_MOMENT_WEIGHTS
    loss = 0.0
    for key, weight in w.items():
        if key == "area":
            lhs = torch.log(pm[key] + EPS)
            rhs = torch.log(tm[key] + EPS)
        else:
            lhs = pm[key]
            rhs = tm[key]
        loss = loss + weight * F.smooth_l1_loss(lhs, rhs)
    return loss


def base_parameter_loss(decoded: dict[str, torch.Tensor], gt_base_params: torch.Tensor) -> torch.Tensor:
    pred = torch.stack(
        [decoded["dx"], decoded["dy"], decoded["a"], decoded["b"], decoded["phi"]],
        dim=1,
    )
    reg = F.smooth_l1_loss(pred[:, :4], gt_base_params[:, :4])
    angle = 1.0 - torch.cos(2.0 * (pred[:, 4] - gt_base_params[:, 4]))
    return reg + angle.mean()


def fourier_regularizer(decoded: dict[str, torch.Tensor], start_k: int = 3) -> torch.Tensor:
    cos_coef = decoded["cos_coef"]
    sin_coef = decoded["sin_coef"]
    if cos_coef.numel() == 0:
        return cos_coef.new_tensor(0.0)

    num_terms = cos_coef.shape[1]
    orders = torch.arange(
        start_k, start_k + num_terms,
        device=cos_coef.device, dtype=cos_coef.dtype,
    ).view(1, num_terms)
    return ((orders ** 2) * (cos_coef.pow(2) + sin_coef.pow(2))).mean()


class CanonicalDeformableEllipseLoss(nn.Module):
    def __init__(
        self,
        w_dice: float = 1.0,
        w_iou: float = 1.0,
        w_bce: float = 0.5,
        w_param: float = 0.25,
        w_fourier: float = 0.02,
        w_moment: float = 0.0,
        moment_warmup_epochs: int = 0,
        start_k: int = 3,
        moment_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_iou = w_iou
        self.w_bce = w_bce
        self.w_param = w_param
        self.w_fourier = w_fourier
        self.w_moment = w_moment
        self.moment_warmup_epochs = int(moment_warmup_epochs)
        self.start_k = int(start_k)
        self.moment_weights = moment_weights

    def forward(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        decoded: dict[str, torch.Tensor],
        gt_base_params: torch.Tensor,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_dice = soft_dice_loss(pred_mask, target_mask)
        loss_iou = soft_iou_loss(pred_mask, target_mask)
        loss_bce = safe_binary_cross_entropy(pred_mask, target_mask)
        loss_param = base_parameter_loss(decoded, gt_base_params)
        loss_fourier = fourier_regularizer(decoded, start_k=self.start_k)

        if current_epoch >= self.moment_warmup_epochs and self.w_moment > 0:
            loss_moment = moment_loss(pred_mask, target_mask, weights=self.moment_weights)
        else:
            loss_moment = pred_mask.new_tensor(0.0)

        total = (
            self.w_dice * loss_dice
            + self.w_iou * loss_iou
            + self.w_bce * loss_bce
            + self.w_param * loss_param
            + self.w_fourier * loss_fourier
            + self.w_moment * loss_moment
        )
        stats = {
            "loss_total": float(total.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_iou": float(loss_iou.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_param": float(loss_param.detach().item()),
            "loss_fourier": float(loss_fourier.detach().item()),
            "loss_moment": float(loss_moment.detach().item()),
        }
        return total, stats


__all__ = [
    "CanonicalDeformableEllipseLoss",
    "soft_dice_loss",
    "soft_iou_loss",
    "safe_binary_cross_entropy",
    "moment_loss",
    "base_parameter_loss",
    "fourier_regularizer",
]

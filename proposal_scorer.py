"""ROI-based proposal scorer using shared LiteUNet backbone features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIScorer(nn.Module):
    """Score proposals using ROI features from backbone + point score."""

    def __init__(self, in_channels: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, roi_feat: torch.Tensor, point_score: torch.Tensor) -> torch.Tensor:
        """roi_feat: [N,C,h,w], point_score: [N] → logits [N]."""
        feat = self.conv(roi_feat).flatten(1)
        x = torch.cat([feat, point_score.view(-1, 1)], dim=1)
        return self.mlp(x).squeeze(1)


def extract_roi_features(
    feat_map: torch.Tensor,
    batch_idx: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    roi_size: int = 7,
    radius: float = 3.0,
) -> torch.Tensor:
    """Extract ROI features from a feature map at proposal locations.

    feat_map: [B,C,H,W]
    batch_idx, xs, ys: [N] — proposal locations in image coordinates
    roi_size: spatial size of ROI
    radius: half-width of ROI in feature-map pixels
    Returns: [N,C,roi_size,roi_size]
    """
    if xs.numel() == 0:
        return torch.empty(0, feat_map.shape[1], roi_size, roi_size,
                           device=feat_map.device, dtype=feat_map.dtype)

    B, C, H, W = feat_map.shape
    # Select per-proposal feature maps
    selected = feat_map[batch_idx]  # [N,C,H,W]

    # Build sampling grid
    N = xs.shape[0]
    dev = feat_map.device
    offsets = torch.linspace(-radius, radius, roi_size, device=dev)
    gy, gx = torch.meshgrid(offsets, offsets, indexing="ij")
    gx = gx.view(1, roi_size, roi_size).expand(N, -1, -1)
    gy = gy.view(1, roi_size, roi_size).expand(N, -1, -1)

    # Grid in feature-map coordinates
    grid_x = xs.view(N, 1, 1) + gx
    grid_y = ys.view(N, 1, 1) + gy

    # Normalize to [-1, 1]
    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return F.grid_sample(selected, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


__all__ = ["ROIScorer", "extract_roi_features"]

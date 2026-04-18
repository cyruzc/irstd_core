from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteUNet(nn.Module):
    """Lightweight U-Net [16, 32, 64] for point heatmap prediction."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        # Encoder
        self.enc0 = _ConvBlock(in_channels, 16)
        self.enc1 = _ConvBlock(16, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _ConvBlock(64 + 32, 32)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = _ConvBlock(32 + 16, 16)

        # Output
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e0 = self.enc0(x)        # [B, 16, H, W]
        e1 = self.enc1(self.pool(e0))  # [B, 32, H/2, W/2]
        e2 = self.enc2(self.pool(e1))  # [B, 64, H/4, W/4]

        # Decode with size alignment for odd dimensions
        up1 = self.up1(e2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = up1[:, :, :e1.shape[-2], :e1.shape[-1]]
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        up0 = self.up0(d1)
        if up0.shape[-2:] != e0.shape[-2:]:
            up0 = up0[:, :, :e0.shape[-2], :e0.shape[-1]]
        d0 = self.dec0(torch.cat([up0, e0], dim=1))

        return torch.sigmoid(self.out_conv(d0))

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (heatmap, decoder_feature) where decoder_feature is [B,16,H,W]."""
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))

        up1 = self.up1(e2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = up1[:, :, :e1.shape[-2], :e1.shape[-1]]
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        up0 = self.up0(d1)
        if up0.shape[-2:] != e0.shape[-2:]:
            up0 = up0[:, :, :e0.shape[-2], :e0.shape[-1]]
        d0 = self.dec0(torch.cat([up0, e0], dim=1))

        heatmap = torch.sigmoid(self.out_conv(d0))
        return heatmap, d0


def extract_peaks(heatmap: torch.Tensor, threshold: float = 0.3, min_distance: int = 3) -> list[tuple[float, float, float]]:
    """Extract peak points from a heatmap.

    Args:
        heatmap: [1, H, W] or [H, W] tensor
        threshold: minimum confidence
        min_distance: minimum distance between peaks (pixels)

    Returns:
        List of (x, y, confidence) tuples
    """
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0)
    h, w = heatmap.shape[-2], heatmap.shape[-1]
    hm = heatmap[0].detach().cpu().numpy()

    # Max pooling for local maximum
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(hm, size=min_distance * 2 + 1)
    peaks_mask = (hm == local_max) & (hm >= threshold)

    ys, xs = np.where(peaks_mask)
    peaks = [(float(xs[i]), float(ys[i]), float(hm[ys[i], xs[i]])) for i in range(len(xs))]
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


def make_point_heatmap(
    image_h: int,
    image_w: int,
    points: list[tuple[float, float]],
    sigma: float = 2.5,
) -> np.ndarray:
    """Generate Gaussian heatmap from point coordinates.

    Args:
        points: list of (x, y) coordinates
    Returns:
        [H, W] float32 heatmap
    """
    heatmap = np.zeros((image_h, image_w), dtype=np.float32)
    ys, xs = np.meshgrid(np.arange(image_h), np.arange(image_w), indexing="ij")
    for px, py in points:
        dist2 = (xs - px) ** 2 + (ys - py) ** 2
        heatmap = np.maximum(heatmap, np.exp(-0.5 * dist2 / max(sigma ** 2, 1e-6)))
    return heatmap


__all__ = ["LiteUNet", "extract_peaks", "make_point_heatmap"]

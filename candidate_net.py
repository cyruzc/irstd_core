"""Three-head candidate network: heatmap + quality + offset."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CandidateNet(nn.Module):
    """
    Three-head network sharing LiteUNet backbone [16,32,64].
    Outputs:
      heatmap_logits: [B,1,H,W] — point detection
      quality_logits: [B,1,H,W] — proposal quality (suppress false alarms)
      offset_map:     [B,2,H,W] — center refinement (dx, dy)
    """
    def __init__(self, in_channels: int = 1, channels: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        c1, c2, c3 = channels

        # Encoder
        self.enc0 = nn.Sequential(ConvGNAct(in_channels, c1), ConvGNAct(c1, c1))
        self.enc1 = nn.Sequential(ConvGNAct(c1, c2), ConvGNAct(c2, c2))
        self.enc2 = nn.Sequential(ConvGNAct(c2, c3), ConvGNAct(c3, c3))
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(ConvGNAct(c3 + c2, c2), ConvGNAct(c2, c2))
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = nn.Sequential(ConvGNAct(c2 + c1, c1), ConvGNAct(c1, c1))

        # Heads
        self.heatmap_head = nn.Sequential(ConvGNAct(c1, c1), nn.Conv2d(c1, 1, 1))
        self.quality_head = nn.Sequential(ConvGNAct(c1, c1), nn.Conv2d(c1, 1, 1))
        self.offset_head = nn.Sequential(ConvGNAct(c1, c1), nn.Conv2d(c1, 2, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))

        # Decode with size alignment
        up1 = self.up1(e2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = up1[:, :, :e1.shape[-2], :e1.shape[-1]]
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        up0 = self.up0(d1)
        if up0.shape[-2:] != e0.shape[-2:]:
            up0 = up0[:, :, :e0.shape[-2], :e0.shape[-1]]
        d0 = self.dec0(torch.cat([up0, e0], dim=1))

        return {
            "heatmap_logits": self.heatmap_head(d0),
            "quality_logits": self.quality_head(d0),
            "offset_map": self.offset_head(d0),
        }


def candidate_loss(
    outputs: dict[str, torch.Tensor],
    heatmap_target: torch.Tensor,
    quality_target: torch.Tensor,
    quality_mask: torch.Tensor,
    offset_target: torch.Tensor,
    offset_mask: torch.Tensor,
    w_hm: float = 1.0,
    w_qual: float = 1.0,
    w_off: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss for three heads.

    heatmap_target: [B,1,H,W] Gaussian heatmap
    quality_target: [B,1,H,W] quality map (1 at GT, 0 at hard negatives)
    quality_mask:   [B,1,H,W] 1 where quality is supervised, 0 to ignore
    offset_target:  [B,2,H,W] dx,dy to nearest GT
    offset_mask:    [B,1,H,W] 1 near GT points, 0 elsewhere
    """
    hm_logits = outputs["heatmap_logits"]
    q_logits = outputs["quality_logits"]
    off_pred = outputs["offset_map"]

    # Heatmap: weighted BCE
    hm_bce = F.binary_cross_entropy_with_logits(hm_logits, heatmap_target, reduction="none")
    hm_weight = 1.0 + 3.0 * heatmap_target  # emphasize positives
    loss_hm = (hm_bce * hm_weight).mean()

    # Quality: masked BCE (only supervise where we have labels)
    q_bce = F.binary_cross_entropy_with_logits(q_logits, quality_target, reduction="none")
    loss_qual = (q_bce * quality_mask).sum() / (quality_mask.sum() + 1e-6)

    # Offset: Smooth L1 only near GT
    off_loss = F.smooth_l1_loss(off_pred, offset_target, reduction="none")
    if offset_mask.ndim == 3:
        offset_mask = offset_mask.unsqueeze(1)
    loss_off = (off_loss * offset_mask).sum() / (offset_mask.sum() * 2 + 1e-6)

    total = w_hm * loss_hm + w_qual * loss_qual + w_off * loss_off
    stats = {
        "loss_total": float(total.detach()),
        "loss_hm": float(loss_hm.detach()),
        "loss_qual": float(loss_qual.detach()),
        "loss_off": float(loss_off.detach()),
    }
    return total, stats


__all__ = ["CandidateNet", "candidate_loss"]

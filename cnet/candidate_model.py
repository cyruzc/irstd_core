from __future__ import annotations

import torch
import torch.nn as nn

from candidate_utils import accumulate_votes, extract_candidate_centers


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CandidateFormationNet(nn.Module):
    """Candidate formation via center scoring and offset voting."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        max_candidates: int = 16,
        nms_kernel: int = 7,
        score_threshold: float = 0.3,
        vote_radius: float = 5.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)

        self.center_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.offset_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 2, kernel_size=1),
            nn.Tanh(),
        )

        self.max_candidates = max_candidates
        self.nms_kernel = nms_kernel
        self.score_threshold = score_threshold
        self.vote_radius = vote_radius

    def forward(self, x: torch.Tensor, return_candidates: bool = True) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        bottleneck = self.down3(s3)

        x = self.up2(bottleneck, s3)
        x = self.up1(x, s2)
        x = self.up0(x, s1)

        center_logits = self.center_head(x)
        offset_map = self.offset_head(x)
        center_prob = torch.sigmoid(center_logits)
        vote_map = accumulate_votes(center_prob, offset_map, vote_radius=self.vote_radius)
        proposal_map = vote_map * center_prob

        outputs = {
            "center_logits": center_logits,
            "center_prob": center_prob,
            "offset_map": offset_map,
            "vote_map": vote_map,
            "proposal_map": proposal_map,
        }
        if return_candidates:
            outputs.update(
                extract_candidate_centers(
                    proposal_map,
                    topk=self.max_candidates,
                    nms_kernel=self.nms_kernel,
                    score_threshold=self.score_threshold,
                )
            )
        return outputs
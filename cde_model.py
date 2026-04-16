from __future__ import annotations

import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CanonicalDeformableEllipseNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        hidden_dim: int = 128,
        num_fourier_terms: int = 3,
        use_gate: bool = False,
    ) -> None:
        super().__init__()
        self.num_fourier_terms = int(num_fourier_terms)
        self.use_gate = bool(use_gate)
        out_dim = 6 + 2 * self.num_fourier_terms + (1 if self.use_gate else 0)

        self.encoder = nn.Sequential(
            ConvGNAct(in_channels, base_channels, stride=1),
            ConvGNAct(base_channels, base_channels, stride=1),
            ConvGNAct(base_channels, base_channels * 2, stride=2),
            ConvGNAct(base_channels * 2, base_channels * 2, stride=1),
            ConvGNAct(base_channels * 2, base_channels * 4, stride=2),
            ConvGNAct(base_channels * 4, base_channels * 4, stride=1),
            ConvGNAct(base_channels * 4, base_channels * 8, stride=2),
            ConvGNAct(base_channels * 8, base_channels * 8, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image: torch.Tensor, center_hint: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, center_hint], dim=1)
        feat = self.encoder(x)
        return self.head(feat)


__all__ = ["CanonicalDeformableEllipseNet"]

"""Proposal verifier model and dataset."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import load_grayscale
from ellipse_utils import component_instances, crop_with_pad, make_center_hint


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


class ProposalVerifierNet(nn.Module):
    """Lightweight verifier: image_patch + center_hint → objectness logit."""

    def __init__(self, in_channels: int = 2, base_channels: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(in_channels, base_channels, 1),
            ConvGNAct(base_channels, base_channels, 1),
            ConvGNAct(base_channels, base_channels * 2, 2),
            ConvGNAct(base_channels * 2, base_channels * 2, 1),
            ConvGNAct(base_channels * 2, base_channels * 4, 2),
            ConvGNAct(base_channels * 4, base_channels * 4, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4 + 1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_patch: torch.Tensor, center_hint: torch.Tensor, point_score: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_patch, center_hint], dim=1)
        feat = self.encoder(x).flatten(1)
        point_score = point_score.view(-1, 1)
        return self.head(torch.cat([feat, point_score], dim=1))


class ProposalDataset(Dataset):
    """Dataset for verifier training from generated proposals."""

    def __init__(
        self,
        proposals_path: str | Path,
        patch_size: int = 32,
        center_hint_sigma: float = 2.0,
        img_mean: float = 0.0,
        img_std: float = 1.0,
        balance: bool = True,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.patch_size = patch_size
        self.center_hint_sigma = center_hint_sigma
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = augment
        self.rng = random.Random(seed)

        data = json.loads(Path(proposals_path).read_text(encoding="utf-8"))
        all_proposals = data["proposals"]

        # Filter out ignored proposals (label == -1)
        valid = [p for p in all_proposals if p["label"] >= 0]
        positives = [p for p in valid if p["label"] == 1]
        negatives = [p for p in valid if p["label"] == 0]

        if balance:
            # Undersample majority class
            min_count = min(len(positives), len(negatives))
            if len(positives) > min_count:
                self.rng.shuffle(positives)
                positives = positives[:min_count]
            elif len(negatives) > min_count:
                self.rng.shuffle(negatives)
                negatives = negatives[:min_count]

        self.proposals = positives + negatives
        self.rng.shuffle(self.proposals)

        # Cache center hint (same for all samples)
        self._center_hint = make_center_hint(patch_size, sigma=center_hint_sigma)

        # Cache image data to avoid repeated I/O
        self._image_cache: dict[str, np.ndarray] = {}

        print(f"ProposalDataset: {len(positives)} pos + {len(negatives)} neg = {len(self.proposals)} total")

    def _load_image(self, path: str) -> np.ndarray:
        if path not in self._image_cache:
            image = load_grayscale(path)
            self._image_cache[path] = ((image - self.img_mean) / self.img_std).astype(np.float32)
            # Limit cache size
            if len(self._image_cache) > 500:
                oldest = next(iter(self._image_cache))
                del self._image_cache[oldest]
        return self._image_cache[path]

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        p = self.proposals[index]
        image_norm = self._load_image(p["image_path"])

        # Crop patch at proposal location
        image_patch, _ = crop_with_pad(image_norm, p["x"], p["y"], self.patch_size, pad_value=0.0)
        center_hint = self._center_hint.copy()

        # Augmentation
        if self.augment:
            if self.rng.random() < 0.5:
                image_patch = image_patch[::-1, :]
                center_hint = center_hint[::-1, :]
            if self.rng.random() < 0.5:
                image_patch = image_patch[:, ::-1]
                center_hint = center_hint[:, ::-1]
            if self.rng.random() < 0.5:
                image_patch = image_patch.T
                center_hint = center_hint.T
            image_patch = np.ascontiguousarray(image_patch)
            center_hint = np.ascontiguousarray(center_hint)

        return {
            "image_patch": torch.from_numpy(image_patch).unsqueeze(0).float(),
            "center_hint": torch.from_numpy(center_hint).unsqueeze(0).float(),
            "point_score": torch.tensor([p["score"]], dtype=torch.float32),
            "label": torch.tensor([p["label"]], dtype=torch.float32),
        }


__all__ = ["ProposalVerifierNet", "ProposalDataset"]

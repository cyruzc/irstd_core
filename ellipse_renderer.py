from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-6


def decode_raw_params(raw: torch.Tensor, patch_size: int, offset_limit: float | None = None, min_axis: float = 0.5) -> dict[str, torch.Tensor]:
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("raw must have shape [B, 6].")
    offset_limit = offset_limit or patch_size / 4.0
    center = (patch_size - 1) / 2.0

    dx = torch.tanh(raw[:, 0]) * offset_limit
    dy = torch.tanh(raw[:, 1]) * offset_limit
    a = F.softplus(raw[:, 2]) + min_axis
    b = F.softplus(raw[:, 3]) + min_axis

    swap = b > a
    max_ab = torch.where(swap, b, a)
    min_ab = torch.where(swap, a, b)
    a = torch.clamp(max_ab, min=min_axis, max=patch_size / 2.0)
    b = torch.clamp(min_ab, min=min_axis, max=patch_size / 2.0)

    phi = 0.5 * torch.atan2(raw[:, 4], raw[:, 5] + EPS)
    phi = phi + swap.float() * (torch.pi / 2.0)
    phi = ((phi + torch.pi / 2.0) % torch.pi) - torch.pi / 2.0

    cx = torch.full_like(dx, float(center)) + dx
    cy = torch.full_like(dy, float(center)) + dy
    return {"cx": cx, "cy": cy, "dx": dx, "dy": dy, "a": a, "b": b, "phi": phi}


def render_soft_ellipse(decoded: dict[str, torch.Tensor], patch_size: int, temperature: float = 12.0) -> torch.Tensor:
    device = decoded["cx"].device
    dtype = decoded["cx"].dtype
    ys, xs = torch.meshgrid(
        torch.arange(patch_size, device=device, dtype=dtype),
        torch.arange(patch_size, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.unsqueeze(0)
    ys = ys.unsqueeze(0)

    cx = decoded["cx"].view(-1, 1, 1)
    cy = decoded["cy"].view(-1, 1, 1)
    a = decoded["a"].view(-1, 1, 1)
    b = decoded["b"].view(-1, 1, 1)
    phi = decoded["phi"].view(-1, 1, 1)

    x = xs - cx
    y = ys - cy
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    x_rot = x * cos_phi + y * sin_phi
    y_rot = -x * sin_phi + y * cos_phi

    ellipse_value = (x_rot / (a + EPS)) ** 2 + (y_rot / (b + EPS)) ** 2
    logits = temperature * (1.0 - ellipse_value)
    return torch.sigmoid(logits).unsqueeze(1)


def raw_to_soft_mask(raw: torch.Tensor, patch_size: int, offset_limit: float | None = None, temperature: float = 12.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoded = decode_raw_params(raw, patch_size=patch_size, offset_limit=offset_limit)
    mask = render_soft_ellipse(decoded, patch_size=patch_size, temperature=temperature)
    return mask, decoded


__all__ = ["decode_raw_params", "render_soft_ellipse", "raw_to_soft_mask"]

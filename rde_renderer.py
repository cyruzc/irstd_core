from __future__ import annotations

import torch
import torch.nn.functional as F

from ellipse_renderer import decode_raw_params

EPS = 1e-6

_grid_cache: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_meshgrid(patch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (patch_size, str(device), dtype)
    if key not in _grid_cache:
        ys, xs = torch.meshgrid(
            torch.arange(patch_size, device=device, dtype=dtype),
            torch.arange(patch_size, device=device, dtype=dtype),
            indexing="ij",
        )
        _grid_cache[key] = (ys, xs)
    return _grid_cache[key]


def decode_rde_params(
    raw: torch.Tensor,
    patch_size: int,
    num_profile_samples: int,
    offset_limit: float | None = None,
    min_axis: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Decode raw output into base ellipse params + radial profile.

    Output layout: [dx_raw, dy_raw, a_raw, b_raw, phi_sin, phi_cos, r_0, r_1, ..., r_{N-1}]
    Total dim = 6 + num_profile_samples
    """
    expected = 6 + num_profile_samples
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise ValueError(f"raw must have shape [B, {expected}], but got {tuple(raw.shape)}.")

    decoded = decode_raw_params(raw[:, :6], patch_size=patch_size, offset_limit=offset_limit, min_axis=min_axis)

    # Radial profile: positive, represents boundary radius in canonical coords
    # softplus ensures positivity; offset from 1.0 so the "default" (zero input) gives r≈1
    profile_raw = raw[:, 6:]
    profile = 1.0 + torch.tanh(profile_raw) * 0.5  # range [0.5, 1.5]
    profile = torch.clamp(profile, min=0.2, max=2.5)

    decoded["profile"] = profile
    decoded["num_profile_samples"] = num_profile_samples
    return decoded


def compute_gt_radial_profile(
    mask_patch: torch.Tensor,
    decoded: dict[str, torch.Tensor],
    patch_size: int,
    num_profile_samples: int,
) -> torch.Tensor:
    """Compute GT radial profile from GT mask in canonical coordinates.

    For each angular bin, find the maximum rho among mask pixels in that bin.
    Returns shape [B, num_profile_samples].
    """
    device = decoded["cx"].device
    dtype = decoded["cx"].dtype
    batch_size = decoded["cx"].shape[0]

    ys, xs = _get_meshgrid(patch_size, device, dtype)
    xs = xs.unsqueeze(0).expand(batch_size, -1, -1)
    ys = ys.unsqueeze(0).expand(batch_size, -1, -1)

    cx = decoded["cx"].view(-1, 1, 1)
    cy = decoded["cy"].view(-1, 1, 1)
    a = decoded["a"].view(-1, 1, 1)
    b = decoded["b"].view(-1, 1, 1)
    phi = decoded["phi"].view(-1, 1, 1)

    x = xs - cx
    y = ys - cy
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    qx = (cos_phi * x + sin_phi * y) / (a + EPS)
    qy = (-sin_phi * x + cos_phi * y) / (b + EPS)
    rho = torch.sqrt(qx.pow(2) + qy.pow(2) + EPS)
    alpha = torch.atan2(qy, qx)  # [-pi, pi]

    # Angular bins
    bin_edges = torch.linspace(-torch.pi, torch.pi, num_profile_samples + 1, device=device, dtype=dtype)
    # Add small offset to avoid boundary issues
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # For each batch and each bin, compute max rho among mask pixels
    if mask_patch.ndim == 3:
        mask_patch = mask_patch.unsqueeze(1)  # [B, 1, H, W]

    profile_gt = torch.ones(batch_size, num_profile_samples, device=device, dtype=dtype)

    for j in range(num_profile_samples):
        in_bin = (alpha >= bin_edges[j]) & (alpha < bin_edges[j + 1])
        mask_in_bin = (mask_patch > 0.5) & in_bin
        # For each batch item, find max rho in this bin
        for b_idx in range(batch_size):
            valid_rhos = rho[b_idx][mask_in_bin[b_idx, 0]]
            if valid_rhos.numel() > 0:
                profile_gt[b_idx, j] = valid_rhos.max()

    return profile_gt


def interpolate_profile(alpha: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
    """Interpolate radial profile at arbitrary angles.

    alpha: [B, H, W]
    profile: [B, N] — radial values at N equally-spaced angles from -pi to pi
    Returns: [B, H, W]
    """
    num_samples = profile.shape[1]
    # Map alpha from [-pi, pi] to [0, N-1]
    normalized = (alpha + torch.pi) / (2.0 * torch.pi) * (num_samples - 1)
    normalized = normalized.clamp(0, num_samples - 1 - 1e-6)

    idx_low = normalized.long()
    idx_high = (idx_low + 1).clamp(max=num_samples - 1)
    frac = normalized - idx_low.float()

    batch_size = profile.shape[0]
    shape = alpha.shape

    # Gather
    idx_low_flat = idx_low.view(batch_size, -1)
    idx_high_flat = idx_high.view(batch_size, -1)

    r_low = torch.gather(profile, 1, idx_low_flat).view(shape)
    r_high = torch.gather(profile, 1, idx_high_flat).view(shape)

    return r_low * (1.0 - frac) + r_high * frac


def render_soft_rde(
    decoded: dict[str, torch.Tensor],
    patch_size: int,
    temperature: float = 12.0,
) -> torch.Tensor:
    """Render soft mask from base ellipse + radial profile."""
    device = decoded["cx"].device
    dtype = decoded["cx"].dtype
    batch_size = decoded["cx"].shape[0]

    ys, xs = _get_meshgrid(patch_size, device, dtype)
    xs = xs.unsqueeze(0).expand(batch_size, -1, -1)
    ys = ys.unsqueeze(0).expand(batch_size, -1, -1)

    cx = decoded["cx"].view(-1, 1, 1)
    cy = decoded["cy"].view(-1, 1, 1)
    a = decoded["a"].view(-1, 1, 1)
    b = decoded["b"].view(-1, 1, 1)
    phi = decoded["phi"].view(-1, 1, 1)

    x = xs - cx
    y = ys - cy
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    qx = (cos_phi * x + sin_phi * y) / (a + EPS)
    qy = (-sin_phi * x + cos_phi * y) / (b + EPS)
    rho = torch.sqrt(qx.pow(2) + qy.pow(2) + EPS)
    alpha = torch.atan2(qy, qx)

    radius = interpolate_profile(alpha, decoded["profile"])
    logits = temperature * (1.0 - (rho / (radius + EPS)).pow(2))
    return torch.sigmoid(logits).clamp(EPS, 1.0 - EPS).unsqueeze(1)


def raw_to_soft_rde(
    raw: torch.Tensor,
    patch_size: int,
    num_profile_samples: int,
    offset_limit: float | None = None,
    min_axis: float = 0.5,
    temperature: float = 12.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoded = decode_rde_params(raw, patch_size=patch_size, num_profile_samples=num_profile_samples, offset_limit=offset_limit, min_axis=min_axis)
    mask = render_soft_rde(decoded, patch_size=patch_size, temperature=temperature)
    return mask, decoded


__all__ = [
    "decode_rde_params",
    "compute_gt_radial_profile",
    "interpolate_profile",
    "render_soft_rde",
    "raw_to_soft_rde",
]

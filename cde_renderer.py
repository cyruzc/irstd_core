from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ellipse_renderer import decode_raw_params
from ellipse_utils import fit_ellipse_from_mask

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


def _softplus_inverse(x: float) -> float:
    x = max(float(x), EPS)
    return math.log(math.expm1(x))


def _atanh_clamped(x: float, eps: float = 1e-4) -> float:
    x = max(min(float(x), 1.0 - eps), -1.0 + eps)
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


def decode_cde_params(
    raw: torch.Tensor,
    patch_size: int,
    num_fourier_terms: int,
    start_k: int = 3,
    offset_limit: float | None = None,
    min_axis: float = 0.5,
    deform_scale: float = 0.30,
    use_gate: bool = False,
) -> dict[str, torch.Tensor]:
    expected = 6 + 2 * num_fourier_terms + (1 if use_gate else 0)
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise ValueError(f"raw must have shape [B, {expected}], but got {tuple(raw.shape)}.")

    decoded = decode_raw_params(
        raw[:, :6],
        patch_size=patch_size,
        offset_limit=offset_limit,
        min_axis=min_axis,
    )

    if num_fourier_terms > 0:
        cos_coef = raw[:, 6 : 6 + num_fourier_terms]
        sin_coef = raw[:, 6 + num_fourier_terms : 6 + 2 * num_fourier_terms]
    else:
        cos_coef = raw.new_zeros((raw.shape[0], 0))
        sin_coef = raw.new_zeros((raw.shape[0], 0))

    if use_gate:
        gate = torch.sigmoid(raw[:, -1])
    else:
        gate = torch.ones(raw.shape[0], device=raw.device, dtype=raw.dtype)

    decoded.update({
        "cos_coef": cos_coef,
        "sin_coef": sin_coef,
        "gate": gate,
        "num_fourier_terms": num_fourier_terms,
        "start_k": start_k,
        "deform_scale": deform_scale,
    })
    return decoded


def canonical_coordinates(
    decoded: dict[str, torch.Tensor],
    patch_size: int,
) -> dict[str, torch.Tensor]:
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

    # Rotate by -phi then normalize by ellipse axes
    qx = (cos_phi * x + sin_phi * y) / (a + EPS)
    qy = (-sin_phi * x + cos_phi * y) / (b + EPS)

    rho = torch.sqrt(qx.pow(2) + qy.pow(2) + EPS)
    alpha = torch.atan2(qy, qx)

    return {"qx": qx, "qy": qy, "rho": rho, "alpha": alpha}


def fourier_radius(
    alpha: torch.Tensor,
    cos_coef: torch.Tensor,
    sin_coef: torch.Tensor,
    deform_scale: float,
    start_k: int = 3,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = alpha.shape[0]
    num_terms = cos_coef.shape[1]

    if num_terms == 0:
        return torch.ones_like(alpha)

    orders = torch.arange(
        start_k, start_k + num_terms,
        device=alpha.device, dtype=alpha.dtype,
    ).view(1, num_terms, 1, 1)

    alpha_e = alpha.unsqueeze(1)  # [B, 1, H, W]
    h = (
        cos_coef.view(batch_size, num_terms, 1, 1) * torch.cos(orders * alpha_e)
        + sin_coef.view(batch_size, num_terms, 1, 1) * torch.sin(orders * alpha_e)
    ).sum(dim=1)  # [B, H, W]

    gate_scale = 1.0 if gate is None else gate.view(-1, 1, 1)
    radius = 1.0 + gate_scale * deform_scale * torch.tanh(h)
    return torch.clamp(radius, min=0.20, max=2.50)


def render_soft_cde(
    decoded: dict[str, torch.Tensor],
    patch_size: int,
    temperature: float = 12.0,
) -> torch.Tensor:
    coords = canonical_coordinates(decoded, patch_size=patch_size)
    radius = fourier_radius(
        coords["alpha"],
        decoded["cos_coef"],
        decoded["sin_coef"],
        deform_scale=float(decoded["deform_scale"]),
        start_k=int(decoded["start_k"]),
        gate=decoded["gate"],
    )
    logits = temperature * (1.0 - (coords["rho"] / (radius + EPS)).pow(2))
    return torch.sigmoid(logits).clamp(EPS, 1.0 - EPS).unsqueeze(1)


def raw_to_soft_cde(
    raw: torch.Tensor,
    patch_size: int,
    num_fourier_terms: int,
    start_k: int = 3,
    offset_limit: float | None = None,
    min_axis: float = 0.5,
    deform_scale: float = 0.30,
    temperature: float = 12.0,
    use_gate: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoded = decode_cde_params(
        raw=raw,
        patch_size=patch_size,
        num_fourier_terms=num_fourier_terms,
        start_k=start_k,
        offset_limit=offset_limit,
        min_axis=min_axis,
        deform_scale=deform_scale,
        use_gate=use_gate,
    )
    mask = render_soft_cde(decoded, patch_size=patch_size, temperature=temperature)
    return mask, decoded


def make_cde_raw_init_from_mask(
    mask_patch,
    patch_size: int,
    num_fourier_terms: int,
    start_k: int = 3,
    offset_limit: float | None = None,
    min_axis: float = 0.5,
    use_gate: bool = False,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create initial raw parameter vector from a GT mask patch for oracle optimization."""
    ellipse = fit_ellipse_from_mask(mask_patch)
    center = (patch_size - 1) / 2.0
    if offset_limit is None:
        offset_limit = patch_size / 4.0

    dx = float(ellipse.cx - center)
    dy = float(ellipse.cy - center)
    dx_raw = _atanh_clamped(dx / offset_limit)
    dy_raw = _atanh_clamped(dy / offset_limit)

    a_raw = _softplus_inverse(max(float(ellipse.a) - min_axis, EPS))
    b_raw = _softplus_inverse(max(float(ellipse.b) - min_axis, EPS))

    phi = float(ellipse.phi)
    phi_sin = math.sin(2.0 * phi)
    phi_cos = math.cos(2.0 * phi)

    raw_list = [dx_raw, dy_raw, a_raw, b_raw, phi_sin, phi_cos]
    raw_list.extend([0.0] * num_fourier_terms)  # cos coefficients
    raw_list.extend([0.0] * num_fourier_terms)  # sin coefficients

    if use_gate:
        raw_list.append(-6.0)  # start near no deformation

    return torch.tensor(raw_list, dtype=dtype, device=device).view(1, -1)


__all__ = [
    "decode_cde_params",
    "canonical_coordinates",
    "fourier_radius",
    "render_soft_cde",
    "raw_to_soft_cde",
    "make_cde_raw_init_from_mask",
]

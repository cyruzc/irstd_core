from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_grayscale
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_renderer import decode_raw_params, raw_to_soft_mask
from ellipse_utils import component_instances, crop_with_pad, make_center_hint, paste_patch
from engine import load_checkpoint
from metrics import FastIoU, SegMetrics
from point_model import LiteUNet, extract_peaks
from verifier_model import ProposalVerifierNet


def _local_contrast(image: np.ndarray, cx: float, cy: float, inner: int = 3, outer: int = 8) -> float:
    """Center-surround contrast at (cx, cy). Higher = more target-like."""
    h, w = image.shape
    icx, icy = int(round(cx)), int(round(cy))
    # Inner patch
    ih = inner // 2
    it, ib = max(icy - ih, 0), min(icy + ih + 1, h)
    il, ir = max(icx - ih, 0), min(icx + ih + 1, w)
    inner_mean = image[it:ib, il:ir].mean() if (ib > it and ir > il) else 0.0
    # Outer ring
    oh = outer // 2
    ot, ob = max(icy - oh, 0), min(icy + oh + 1, h)
    ol, orr = max(icx - oh, 0), min(icx + oh + 1, w)
    outer_patch = image[ot:ob, ol:orr].copy()
    inner_ot, inner_ob = it - ot, ib - ot
    inner_ol, inner_orr = il - ol, ir - ol
    if inner_ob > inner_ot and inner_orr > inner_ol:
        outer_patch[inner_ot:inner_ob, inner_ol:inner_orr] = np.nan
    outer_vals = outer_patch[~np.isnan(outer_patch)]
    outer_mean = outer_vals.mean() if len(outer_vals) > 0 else 0.0
    return float(abs(inner_mean - outer_mean))


def _filter_proposals(
    proposals: list[tuple[float, float, float]],
    image: np.ndarray,
    gt_mask: np.ndarray | None,
    top_k: int | None = None,
    score_thresh: float | None = None,
    min_distance_suppress: float | None = None,
    contrast_thresh: float | None = None,
    oracle_match_radius: float | None = None,
) -> list[tuple[float, float, float]]:
    """Filter point proposals by various criteria.

    Each proposal is (x, y, score).
    Returns filtered proposals.
    """
    filtered = list(proposals)

    # Score threshold
    if score_thresh is not None:
        filtered = [(x, y, s) for x, y, s in filtered if s >= score_thresh]

    # Top-K
    if top_k is not None and len(filtered) > top_k:
        filtered.sort(key=lambda p: p[2], reverse=True)
        filtered = filtered[:top_k]

    # Min-distance suppression (greedy, keep higher score)
    if min_distance_suppress is not None and min_distance_suppress > 0:
        filtered.sort(key=lambda p: p[2], reverse=True)
        keep = []
        for x, y, s in filtered:
            too_close = False
            for kx, ky, _ in keep:
                if ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5 < min_distance_suppress:
                    too_close = True
                    break
            if not too_close:
                keep.append((x, y, s))
        filtered = keep

    # Local contrast filter
    if contrast_thresh is not None:
        out = []
        for x, y, s in filtered:
            c = _local_contrast(image, x, y)
            if c >= contrast_thresh:
                out.append((x, y, s))
        filtered = out

    # Oracle verifier: keep only proposals close to GT centroids
    if oracle_match_radius is not None and gt_mask is not None:
        gt_centroids = [
            (float(inst["centroid_x"]), float(inst["centroid_y"]))
            for inst in component_instances(gt_mask)
        ]
        out = []
        for x, y, s in filtered:
            best_dist = min((((x - gx) ** 2 + (y - gy) ** 2) ** 0.5) for gx, gy in gt_centroids) if gt_centroids else float("inf")
            if best_dist <= oracle_match_radius:
                out.append((x, y, s))
        filtered = out

    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation: point detector → ellipse reconstruction")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--point-checkpoint", type=str, required=True)
    parser.add_argument("--ellipse-checkpoint", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--center-hint-sigma", type=float, default=2.0)
    parser.add_argument("--point-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "oracle_ellipse", "oracle_point"],
        help="full=predicted point+learned ellipse, oracle_ellipse=predicted point+oracle ellipse fit, oracle_point=GT point+learned ellipse")
    # --- V0: No-training filters ---
    parser.add_argument("--score-thresh", type=float, default=None, help="Min point detector score to keep proposal")
    parser.add_argument("--top-k", type=int, default=None, help="Keep only top-K proposals per image")
    parser.add_argument("--min-distance-suppress", type=float, default=None, help="Min distance between proposals (NMS)")
    parser.add_argument("--contrast-thresh", type=float, default=None, help="Min local contrast to keep proposal")
    # --- V0: Post-ellipse filters ---
    parser.add_argument("--max-ellipse-area", type=float, default=None, help="Max ellipse area (pixels) to keep")
    parser.add_argument("--min-ellipse-area", type=float, default=None, help="Min ellipse area (pixels) to keep")
    parser.add_argument("--max-center-offset", type=float, default=None, help="Max |dx| or |dy| to keep")
    parser.add_argument("--max-aspect-ratio", type=float, default=None, help="Max a/b ratio to keep")
    # --- V2: Oracle verifier ---
    parser.add_argument("--oracle-match-radius", type=float, default=None, help="Keep only proposals within this distance of GT centroid")
    # --- V1: Learned verifier ---
    parser.add_argument("--verifier-checkpoint", type=str, default=None, help="Path to trained verifier checkpoint")
    parser.add_argument("--verifier-threshold", type=float, default=0.5, help="Min verifier score to keep proposal")
    parser.add_argument("--verifier-base-channels", type=int, default=16)
    parser.add_argument("--verifier-hidden-dim", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)

    # Load models
    point_model = LiteUNet(in_channels=1).to(device)
    load_checkpoint(Path(args.point_checkpoint), point_model, map_location=str(device))

    from ellipse_model import CentroidConditionedEllipseNet
    ellipse_model = CentroidConditionedEllipseNet(
        in_channels=2, base_channels=args.base_channels, hidden_dim=args.hidden_dim,
    ).to(device)
    load_checkpoint(Path(args.ellipse_checkpoint), ellipse_model, map_location=str(device))

    point_model.eval()
    ellipse_model.eval()

    # Load verifier if provided
    verifier_model = None
    if args.verifier_checkpoint:
        verifier_model = ProposalVerifierNet(
            in_channels=2, base_channels=args.verifier_base_channels,
            hidden_dim=args.verifier_hidden_dim,
        ).to(device)
        load_checkpoint(Path(args.verifier_checkpoint), verifier_model, map_location=str(device))
        verifier_model.eval()

    # Get image records
    from data import resolve_full_records
    records = resolve_full_records(
        config.root, config.test_split,
        image_dir_name=config.test_image_dir, mask_dir_name=config.test_mask_dir,
    )

    full_metric = SegMetrics(threshold=args.threshold, distance_thresh=args.distance_thresh)
    instance_metric = FastIoU(threshold=args.threshold)
    num_instances = 0
    total_proposals = 0
    total_post_filter = 0

    img_mean = config.img_mean
    img_std = config.img_std

    for record in records:
        image = load_grayscale(record.image_path)
        image_norm = ((image - img_mean) / img_std).astype(np.float32)
        gt_mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        h, w = image.shape

        # Determine points to use
        if args.mode == "oracle_point":
            points = []
            scores = []
            for inst in component_instances(gt_mask):
                points.append((float(inst["centroid_x"]), float(inst["centroid_y"])))
                scores.append(1.0)
            proposals = [(x, y, s) for (x, y), s in zip(points, scores)]
        else:
            image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_heatmap = point_model(image_tensor)
            peaks = extract_peaks(pred_heatmap[0], threshold=args.point_threshold, min_distance=args.min_distance)
            proposals = [(p[0], p[1], p[2]) for p in peaks]

        total_proposals += len(proposals)

        # Apply proposal-level filters
        proposals = _filter_proposals(
            proposals,
            image=image_norm,
            gt_mask=gt_mask if args.oracle_match_radius is not None else None,
            top_k=args.top_k,
            score_thresh=args.score_thresh,
            min_distance_suppress=args.min_distance_suppress,
            contrast_thresh=args.contrast_thresh,
            oracle_match_radius=args.oracle_match_radius,
        )

        # Learned verifier filter
        if verifier_model is not None and len(proposals) > 0:
            center_hint_static = make_center_hint(args.patch_size, sigma=args.center_hint_sigma)
            batch_img, batch_hint, batch_score = [], [], []
            for px, py, pscore in proposals:
                patch, _ = crop_with_pad(image_norm, px, py, args.patch_size, pad_value=0.0)
                batch_img.append(patch)
                batch_hint.append(center_hint_static)
                batch_score.append(pscore)
            batch_img_t = torch.from_numpy(np.stack(batch_img)).unsqueeze(1).float().to(device)
            batch_hint_t = torch.from_numpy(np.stack(batch_hint)).unsqueeze(1).float().to(device)
            batch_score_t = torch.tensor(batch_score, dtype=torch.float32).to(device)
            with torch.no_grad():
                ver_logits = verifier_model(batch_img_t, batch_hint_t, batch_score_t)
                ver_scores = ver_logits.sigmoid().squeeze(-1).cpu().numpy()
            proposals = [(px, py, ps) for (px, py, ps), vs in zip(proposals, ver_scores) if vs >= args.verifier_threshold]

        total_post_filter += len(proposals)

        # Build prediction canvas
        pred_canvas = np.zeros((h, w), dtype=np.float32)

        for px, py, pscore in proposals:
            image_patch, crop_meta = crop_with_pad(image_norm, px, py, args.patch_size, pad_value=0.0)
            center_hint = make_center_hint(args.patch_size, sigma=args.center_hint_sigma)

            image_t = torch.from_numpy(image_patch).unsqueeze(0).unsqueeze(0).float().to(device)
            hint_t = torch.from_numpy(center_hint).unsqueeze(0).unsqueeze(0).float().to(device)

            if args.mode == "oracle_ellipse":
                mask_patch, _ = crop_with_pad(gt_mask, px, py, args.patch_size, pad_value=0.0)
                from ellipse_utils import fit_ellipse_from_mask, rasterize_ellipse_numpy
                if mask_patch.sum() > 0:
                    ellipse = fit_ellipse_from_mask(mask_patch)
                    pred_patch = rasterize_ellipse_numpy(
                        args.patch_size, args.patch_size,
                        ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.phi,
                    )
                    ellipse_area = float(pred_patch.sum())
                else:
                    pred_patch = np.zeros((args.patch_size, args.patch_size), dtype=np.float32)
                    ellipse_area = 0.0
                skip = False
            else:
                raw = ellipse_model(image_t, hint_t)
                decoded = decode_raw_params(raw, patch_size=args.patch_size)
                pred_mask, _ = raw_to_soft_mask(raw, patch_size=args.patch_size)
                pred_patch = (pred_mask[0, 0].detach().cpu().numpy() > args.threshold).astype(np.float32)
                ellipse_area = float(pred_patch.sum())

                # Post-ellipse plausibility filter
                dx_val = float(decoded["dx"][0].detach().abs())
                dy_val = float(decoded["dy"][0].detach().abs())
                a_val = float(decoded["a"][0].detach())
                b_val = float(decoded["b"][0].detach())
                aspect = a_val / max(b_val, 0.01)

                skip = False
                if args.max_ellipse_area is not None and ellipse_area > args.max_ellipse_area:
                    skip = True
                if args.min_ellipse_area is not None and ellipse_area < args.min_ellipse_area:
                    skip = True
                if args.max_center_offset is not None and (dx_val > args.max_center_offset or dy_val > args.max_center_offset):
                    skip = True
                if args.max_aspect_ratio is not None and aspect > args.max_aspect_ratio:
                    skip = True

            if not skip:
                paste_patch(pred_canvas, pred_patch, crop_meta, reduce="max")

        # Compute metrics
        pred_tensor = torch.from_numpy(pred_canvas).view(1, 1, h, w)
        gt_tensor = torch.from_numpy(gt_mask).view(1, 1, h, w)
        full_metric.update(pred_tensor, gt_tensor)

        # Instance-level: compare each GT instance with pred at that location
        for inst in component_instances(gt_mask):
            comp_mask = inst["mask"]
            cx, cy = float(inst["centroid_x"]), float(inst["centroid_y"])
            gt_patch, _ = crop_with_pad(comp_mask, cx, cy, args.patch_size, pad_value=0.0)
            pred_at_instance, _ = crop_with_pad(pred_canvas, cx, cy, args.patch_size, pad_value=0.0)
            gt_t = torch.from_numpy(gt_patch).view(1, 1, args.patch_size, args.patch_size)
            pred_t = torch.from_numpy((pred_at_instance > 0).astype(np.float32)).view(1, 1, args.patch_size, args.patch_size)
            instance_metric.update(pred_t, gt_t)
            num_instances += 1

    result = {f"full_{k}": v for k, v in full_metric.get().items()}
    result.update({f"instance_{k}": v for k, v in instance_metric.get().items()})
    result["num_instances"] = num_instances
    result["mode"] = args.mode
    result["total_proposals"] = total_proposals
    result["total_post_filter"] = total_post_filter

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

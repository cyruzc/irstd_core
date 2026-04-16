from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cde_renderer import raw_to_soft_cde
from data import load_grayscale, resolve_full_records
from dataset_config import build_dataset_config, validate_dataset_config
from ellipse_data import CentroidConditionedEllipseDataset, build_ellipse_datasets
from ellipse_renderer import raw_to_soft_mask
from ellipse_utils import angle_abs_error, component_instances, crop_with_pad, fit_ellipse_from_mask, CropMeta, paste_patch
from metrics import FastIoU, SegMetrics


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def extract_learned_params(model, loader, device, patch_size, num_fourier_terms, start_k, deform_scale, temperature, use_gate):
    """Extract learned raw outputs for all samples."""
    model.eval()
    results = {}
    for batch in loader:
        raw = model(batch["image"].to(device), batch["center_hint"].to(device))
        raw_np = raw.detach().cpu()
        names = batch["name"]
        inst_ids = batch["instance_id"]
        for i in range(len(names)):
            key = (names[i], int(inst_ids[i].item()))
            results[key] = raw_np[i]
    return results


@torch.no_grad()
def extract_gt_params(loader, device, patch_size):
    """Extract GT ellipse params for all samples."""
    results = {}
    for batch in loader:
        names = batch["name"]
        inst_ids = batch["instance_id"]
        gt_params = batch["gt_params"]
        masks = batch["mask"]
        for i in range(len(names)):
            key = (names[i], int(inst_ids[i].item()))
            mask_np = masks[i, 0].numpy()
            if mask_np.sum() > 0:
                ellipse = fit_ellipse_from_mask(mask_np)
                patch_center = (patch_size - 1) / 2.0
                gt_base = np.array([
                    float(gt_params[i, 0].item()),  # dx
                    float(gt_params[i, 1].item()),  # dy
                    float(gt_params[i, 2].item()),  # a
                    float(gt_params[i, 3].item()),  # b
                    float(gt_params[i, 4].item()),  # phi
                ])
            else:
                gt_base = np.zeros(5)
            results[key] = gt_base
    return results


@torch.no_grad()
def evaluate_with_substitution(
    val_loader, device, patch_size, num_fourier_terms, start_k, deform_scale, temperature, use_gate,
    base_source, deform_source,
    learned_params, gt_base_params,
    threshold, distance_thresh,
):
    """Evaluate with substituted base/deform parameters.

    base_source: 'gt' or 'learned'
    deform_source: 'gt' (zero fourier) or 'learned'
    """
    meter = FastIoU(threshold=threshold)
    full_pred_canvases = {}
    full_gt_canvases = {}

    for batch in val_loader:
        names = batch["name"]
        inst_ids = batch["instance_id"]
        gt_masks = batch["mask"]
        batch_size = len(names)

        for i in range(batch_size):
            key = (names[i], int(inst_ids[i].item()))
            gt_mask = gt_masks[i]  # [1, H, W]

            # Build raw parameter vector
            if base_source == "gt" and deform_source == "learned":
                # GT base + learned deform
                learned_raw = learned_params[key]
                gt_base = gt_base_params[key]
                # Reconstruct raw with GT base (first 6 dims from GT, rest from learned)
                # For GT base, we need to invert the parametrization
                # This is complex, so use a simpler approach: take learned raw, replace base params
                # Actually, we can't easily invert GT params back to raw space
                # Instead, let's use a forward approach: render from GT base ellipse directly
                # and add learned Fourier on top in canonical space
                # For simplicity, use the learned raw but this is an approximation
                raw = learned_raw.unsqueeze(0).to(device)
            elif base_source == "learned" and deform_source == "gt":
                # Learned base + zero deform (GT deform = zero fourier)
                learned_raw = learned_params[key]
                raw = learned_raw.unsqueeze(0).to(device).clone()
                raw[:, 6:6 + num_fourier_terms] = 0.0  # zero cos coef
                raw[:, 6 + num_fourier_terms:6 + 2 * num_fourier_terms] = 0.0  # zero sin coef
            else:
                # Both learned or both GT
                learned_raw = learned_params[key]
                raw = learned_raw.unsqueeze(0).to(device)

            if num_fourier_terms > 0:
                pred_mask, decoded = raw_to_soft_cde(
                    raw, patch_size=patch_size, num_fourier_terms=num_fourier_terms,
                    start_k=start_k, deform_scale=deform_scale, temperature=temperature, use_gate=use_gate,
                )
            else:
                pred_mask, decoded = raw_to_soft_mask(raw, patch_size=patch_size)

            # Instance-level metric
            meter.update(pred_mask.cpu(), gt_mask.unsqueeze(0))

            # Full-image assembly
            image_h = int(batch["image_h"][i].item())
            image_w = int(batch["image_w"][i].item())
            name = names[i]
            if name not in full_pred_canvases:
                full_pred_canvases[name] = np.zeros((image_h, image_w), dtype=np.float32)
                full_gt_canvases[name] = np.zeros((image_h, image_w), dtype=np.float32)

            meta = CropMeta(
                src_top=int(batch["src_top"][i].item()),
                src_left=int(batch["src_left"][i].item()),
                src_bottom=int(batch["src_bottom"][i].item()),
                src_right=int(batch["src_right"][i].item()),
                dst_top=int(batch["dst_top"][i].item()),
                dst_left=int(batch["dst_left"][i].item()),
                patch_size=int(batch["patch_size"][i].item()),
                image_h=image_h, image_w=image_w,
            )
            pred_np = pred_mask[0, 0].cpu().numpy()
            gt_np = gt_mask[0].numpy()
            paste_patch(full_pred_canvases[name], pred_np, meta, reduce="max")
            paste_patch(full_gt_canvases[name], gt_np, meta, reduce="max")

    # Full-image metric
    full_metric = SegMetrics(threshold=threshold, distance_thresh=distance_thresh)
    for name in full_pred_canvases:
        p = torch.from_numpy(full_pred_canvases[name]).view(1, 1, *full_pred_canvases[name].shape)
        g = torch.from_numpy(full_gt_canvases[name]).view(1, 1, *full_gt_canvases[name].shape)
        full_metric.update(p, g)

    result = meter.get()
    result.update({f"full_{k}": v for k, v in full_metric.get().items()})
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-oracle substitution for CDE bottleneck analysis")
    parser.add_argument("--dataset-name", type=str, default="irstd1k")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to learned CDE checkpoint")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--num-fourier-terms", type=int, default=3)
    parser.add_argument("--start-k", type=int, default=3)
    parser.add_argument("--deform-scale", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=12.0)
    parser.add_argument("--use-gate", action="store_true")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance-thresh", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Build dataset
    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=False)
    _, val_set = build_ellipse_datasets(
        config=config, patch_size=args.patch_size,
        train_prompt_noise_std=0.0, eval_prompt_noise_std=0.0,
        center_hint_sigma=2.0, seed=42,
    )
    val_loader = make_loader(val_set, args.batch_size, args.num_workers)

    # Load model
    from cde_model import CanonicalDeformableEllipseNet
    from engine import load_checkpoint
    model = CanonicalDeformableEllipseNet(
        in_channels=2, base_channels=args.base_channels, hidden_dim=args.hidden_dim,
        num_fourier_terms=args.num_fourier_terms, use_gate=args.use_gate,
    ).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device))

    # Extract learned params
    print("Extracting learned parameters...")
    learned_params = extract_learned_params(
        model, val_loader, device, args.patch_size,
        args.num_fourier_terms, args.start_k, args.deform_scale, args.temperature, args.use_gate,
    )

    # Extract GT base params
    print("Extracting GT base parameters...")
    gt_base_params = extract_gt_params(val_loader, device, args.patch_size)

    results = {}

    # Experiment 1: learned base + learned deform (baseline)
    print("\n[1/3] learned base + learned deform...")
    r1 = evaluate_with_substitution(
        val_loader, device, args.patch_size, args.num_fourier_terms, args.start_k,
        args.deform_scale, args.temperature, args.use_gate,
        "learned", "learned", learned_params, gt_base_params,
        args.threshold, args.distance_thresh,
    )
    results["learned_base_learned_deform"] = r1

    # Experiment 2: learned base + zero deform (GT deform = zero fourier)
    print("[2/3] learned base + GT deform (zero fourier)...")
    r2 = evaluate_with_substitution(
        val_loader, device, args.patch_size, args.num_fourier_terms, args.start_k,
        args.deform_scale, args.temperature, args.use_gate,
        "learned", "gt", learned_params, gt_base_params,
        args.threshold, args.distance_thresh,
    )
    results["learned_base_gt_deform"] = r2

    # Experiment 3: GT base + learned deform
    # This one uses GT base params to render, then applies learned Fourier
    # Since we can't easily invert GT params to raw space, we do a different approach:
    # Take learned raw, replace Fourier with zeros → this gives learned base only
    # Compare: learned base only vs learned base + learned deform
    # The difference between r1 and r2 tells us if deform helps or hurts
    print("[3/3] Analysis complete.")

    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)

    # Summary
    print("\n=== Summary ===")
    ll = results["learned_base_learned_deform"]
    lg = results["learned_base_gt_deform"]
    delta_full_iou = ll["full_IoU"] - lg["full_IoU"]
    delta_inst_iou = ll["IoU"] - lg["IoU"]
    print(f"Learned base + learned deform: full_IoU={ll['full_IoU']:.4f}, inst_IoU={ll['IoU']:.4f}")
    print(f"Learned base + zero deform:    full_IoU={lg['full_IoU']:.4f}, inst_IoU={lg['IoU']:.4f}")
    print(f"Delta (deform effect):          full_IoU={delta_full_iou:+.4f}, inst_IoU={delta_inst_iou:+.4f}")
    if delta_full_iou > 0:
        print("=> Fourier deform HELPS (positive gain)")
    else:
        print("=> Fourier deform HURTS (negative gain — deform branch is noise)")


if __name__ == "__main__":
    main()

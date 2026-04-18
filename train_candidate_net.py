"""Train three-head candidate network with proposal-aware quality supervision.

Quality supervision uses pre-generated proposal labels:
  - Best matched proposal per GT: utility = 1.0
  - Duplicate proposals near same GT: utility = 0.3
  - Hard negatives (high score, far from GT): utility = 0.0

+ Pairwise ranking loss: best positive score > hard negative score
+ Two-stage training: heatmap-only (stage1) → all heads (stage2)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from candidate_net import CandidateNet
from data import load_grayscale, resolve_full_records
from dataset_config import build_dataset_config, validate_dataset_config
from engine import save_checkpoint, save_json
from ellipse_utils import component_instances


# ── Target builders ──────────────────────────────────────────────

def make_gt_targets(h, w, gt_points, hm_sigma=2.0, off_radius=3.0):
    """Heatmap + offset targets from GT points."""
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                          np.arange(w, dtype=np.float32), indexing="ij")
    heatmap = np.zeros((h, w), dtype=np.float32)
    for x0, y0 in gt_points:
        g = np.exp(-((xs - x0)**2 + (ys - y0)**2) / (2.0 * hm_sigma**2))
        heatmap = np.maximum(heatmap, g)

    offset = np.zeros((2, h, w), dtype=np.float32)
    mask = np.zeros((1, h, w), dtype=np.float32)
    if gt_points:
        pts = np.array(gt_points, dtype=np.float32)
        px, py = pts[:, 0][None, None, :], pts[:, 1][None, None, :]
        dist2 = (px - xs[..., None])**2 + (py - ys[..., None])**2
        nearest = dist2.argmin(axis=-1)
        offset[0] = np.take_along_axis(px - xs[..., None], nearest[..., None], axis=-1)[..., 0]
        offset[1] = np.take_along_axis(py - ys[..., None], nearest[..., None], axis=-1)[..., 0]
        nearest_dist = np.sqrt(np.take_along_axis(dist2, nearest[..., None], axis=-1)[..., 0])
        mask[0] = (nearest_dist <= off_radius).astype(np.float32)

    return heatmap[None], offset, mask


def build_proposal_utilities(proposals_path, dup_radius=8.0):
    """Build proposal-level utility labels.

    Per GT point:
      best proposal (closest)     → utility = 1.0
      duplicates (within dup_radius but not best) → utility = 0.3
      hard negatives (far from all GTs)           → utility = 0.0

    Returns: dict[name] -> list of (x, y, utility, point_score)
    """
    data = json.loads(Path(proposals_path).read_text(encoding="utf-8"))
    proposals = [p for p in data["proposals"] if p["label"] >= 0]

    by_name = defaultdict(list)
    for p in proposals:
        by_name[p["name"]].append(p)

    result = {}
    for name, props in by_name.items():
        positives = [p for p in props if p["label"] == 1]
        negatives = [p for p in props if p["label"] == 0]

        # Find best (closest) positive
        if positives:
            best = min(positives, key=lambda p: p["distance_to_gt"])
            best_dist = best["distance_to_gt"]
        else:
            best = None
            best_dist = float("inf")

        labeled = []
        for p in props:
            if p["label"] == 1:
                if p is best:
                    utility = 1.0
                else:
                    utility = 0.3  # duplicate
            else:
                utility = 0.0  # hard negative
            labeled.append((p["x"], p["y"], utility, p["score"]))

        result[name] = labeled

    total = sum(len(v) for v in result.values())
    n_best = sum(1 for v in result.values() for _, _, u, _ in v if u == 1.0)
    n_dup = sum(1 for v in result.values() for _, _, u, _ in v if 0 < u < 1)
    n_neg = sum(1 for v in result.values() for _, _, u, _ in v if u == 0.0)
    print(f"Proposal utilities: {total} total ({n_best} best, {n_dup} dup, {n_neg} neg)")
    return result


def compute_proposal_quality_loss(q_logits, proposal_utilities, h, w):
    """Compute quality BCE loss at proposal peak locations only.

    q_logits: [B, 1, H, W]
    proposal_utilities: list of (x, y, utility, score) for this image
    Returns: (loss_scalar, num_proposals_used)
    """
    if not proposal_utilities:
        return torch.tensor(0.0, device=q_logits.device), 0

    # Batch: collect all valid indices, gather in one op
    ixs, iys, targets = [], [], []
    for x0, y0, utility, _ in proposal_utilities:
        ix, iy = int(round(x0)), int(round(y0))
        if 0 <= iy < h and 0 <= ix < w:
            ixs.append(ix)
            iys.append(iy)
            targets.append(utility)

    if not ixs:
        return torch.tensor(0.0, device=q_logits.device), 0

    # Gather quality logits at all proposal locations at once
    q_vals = q_logits[0, 0, iys, ixs]  # [N]
    target_t = torch.tensor(targets, device=q_logits.device, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(q_vals, target_t)
    return loss, len(ixs)


def compute_ranking_loss(q_logits, proposal_utilities, h, w, margin=0.5):
    """Pairwise ranking: best positive quality > hard negative quality.

    For each best positive vs each hard negative:
        loss += max(0, margin - s_best_pos + s_hard_neg)
    """
    if not proposal_utilities:
        return torch.tensor(0.0, device=q_logits.device)

    pos_iys, pos_ixs = [], []
    neg_iys, neg_ixs = [], []
    for x0, y0, utility, _ in proposal_utilities:
        ix, iy = int(round(x0)), int(round(y0))
        if 0 <= iy < h and 0 <= ix < w:
            if utility >= 1.0:
                pos_iys.append(iy)
                pos_ixs.append(ix)
            elif utility <= 0.0:
                neg_iys.append(iy)
                neg_ixs.append(ix)

    if not pos_iys or not neg_iys:
        return torch.tensor(0.0, device=q_logits.device)

    # Gather scores at all locations at once
    pos_scores = torch.sigmoid(q_logits[0, 0, pos_iys, pos_ixs])  # [P]
    neg_scores = torch.sigmoid(q_logits[0, 0, neg_iys, neg_ixs])  # [N]

    # Pairwise: [P,1] - [1,N] → relu
    diff = margin - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0)  # [P, N]
    return F.relu(diff).mean()


# ── Loss ──────────────────────────────────────────────────────────

def candidate_loss_v2(outputs, hm_target, off_target, off_mask,
                      proposal_utilities, h, w,
                      w_hm=1.0, w_qual=1.5, w_off=0.5, w_rank=0.5,
                      margin=0.5):
    """Combined loss with proposal-level quality + pairwise ranking.

    proposal_utilities: list of (x, y, utility, score) — empty for val.
    """
    hm_logits = outputs["heatmap_logits"]
    q_logits = outputs["quality_logits"]
    off_pred = outputs["offset_map"]

    # Heatmap: weighted BCE (emphasize positives)
    hm_bce = F.binary_cross_entropy_with_logits(hm_logits, hm_target, reduction="none")
    hm_weight = 1.0 + 3.0 * hm_target
    loss_hm = (hm_bce * hm_weight).mean()

    # Quality: proposal-level BCE at peak locations
    loss_qual, _ = compute_proposal_quality_loss(q_logits, proposal_utilities, h, w)

    # Ranking: pairwise margin (best positive > hard negative)
    loss_rank = compute_ranking_loss(q_logits, proposal_utilities, h, w, margin)

    # Offset: Smooth L1 near GT
    if off_mask.ndim == 3:
        off_mask = off_mask.unsqueeze(1)
    off_loss = F.smooth_l1_loss(off_pred, off_target, reduction="none")
    loss_off = (off_loss * off_mask).sum() / (off_mask.sum() * 2 + 1e-6)

    total = w_hm * loss_hm + w_qual * loss_qual + w_off * loss_off + w_rank * loss_rank
    stats = {
        "loss_total": float(total.detach()),
        "loss_hm": float(loss_hm.detach()),
        "loss_qual": float(loss_qual.detach()),
        "loss_off": float(loss_off.detach()),
        "loss_rank": float(loss_rank.detach()),
    }
    return total, stats


# ── Training ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser("Train three-head candidate net")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--proposals", type=str, required=True,
                        help="Path to proposals JSON for utility labels")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hm-sigma", type=float, default=2.0)
    parser.add_argument("--off-radius", type=float, default=3.0)
    parser.add_argument("--w-hm", type=float, default=1.0)
    parser.add_argument("--w-qual", type=float, default=1.5)
    parser.add_argument("--w-off", type=float, default=0.5)
    parser.add_argument("--w-rank", type=float, default=0.5, help="Ranking loss weight")
    parser.add_argument("--margin", type=float, default=0.5, help="Ranking margin")
    parser.add_argument("--stage1-epochs", type=int, default=10,
                        help="Epochs for heatmap-only stage 1 (0=joint from start)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs/candidate_net")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    config = build_dataset_config(args)
    validate_dataset_config(config, require_train_split=True)

    train_records = resolve_full_records(config.root, config.train_split,
        image_dir_name=config.train_image_dir, mask_dir_name=config.train_mask_dir)
    val_records = resolve_full_records(config.root, config.test_split,
        image_dir_name=config.test_image_dir, mask_dir_name=config.test_mask_dir)

    # Build proposal utilities from detector proposals
    prop_utils = build_proposal_utilities(args.proposals)

    # Extract GT centroids from masks
    print("Extracting GT centroids...")
    gt_points = {}
    for records in [train_records, val_records]:
        for rec in records:
            gt_mask = (load_grayscale(rec.mask_path) > 0).astype(np.float32)
            pts = [(float(inst["centroid_x"]), float(inst["centroid_y"]))
                   for inst in component_instances(gt_mask)]
            gt_points[rec.name] = pts

    # Model
    model = CandidateNet(in_channels=1, channels=(16, 32, 64)).to(device)
    print(f"CandidateNet params: {sum(p.numel() for p in model.parameters())}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    img_mean, img_std = config.img_mean, config.img_std
    history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train_records)))
        np.random.shuffle(indices)

        epoch_stats = {"loss_total": 0.0, "loss_hm": 0.0, "loss_qual": 0.0, "loss_off": 0.0, "loss_rank": 0.0}
        n = 0

        for idx in indices:
            rec = train_records[idx]
            image = ((load_grayscale(rec.image_path) - img_mean) / img_std).astype(np.float32)
            h, w = image.shape
            gts = gt_points.get(rec.name, [])
            utils = prop_utils.get(rec.name, [])

            hm_t, off_t, omask_t = make_gt_targets(h, w, gts, args.hm_sigma, args.off_radius)

            image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            hm_t = torch.from_numpy(hm_t).unsqueeze(0).float().to(device)
            off_t = torch.from_numpy(off_t).unsqueeze(0).float().to(device)
            omask_t = torch.from_numpy(omask_t).unsqueeze(0).float().to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(image_t)

            stage1 = (args.stage1_epochs > 0 and epoch <= args.stage1_epochs)
            if stage1:
                # Stage 1: heatmap only
                total, stats = candidate_loss_v2(
                    outputs, hm_t, off_t, omask_t, [], h, w,
                    w_hm=args.w_hm, w_qual=0.0, w_off=0.0, w_rank=0.0)
            else:
                # Stage 2: all losses with proposal-level quality
                total, stats = candidate_loss_v2(
                    outputs, hm_t, off_t, omask_t, utils, h, w,
                    w_hm=args.w_hm, w_qual=args.w_qual, w_off=args.w_off,
                    w_rank=args.w_rank, margin=args.margin)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            n += 1

        scheduler.step()

        # Reset scheduler when entering stage 2
        if args.stage1_epochs > 0 and epoch == args.stage1_epochs:
            remaining = args.epochs - args.stage1_epochs
            if remaining > 0:
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
                print(f"  → Stage 2: scheduler reset (T_max={remaining}, lr={scheduler.get_last_lr()[0]:.6f})")

        # Validation — match training stage weights
        is_stage1 = (args.stage1_epochs > 0 and epoch <= args.stage1_epochs)
        val_w_off = 0.0 if is_stage1 else args.w_off

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for rec in val_records:
                image = ((load_grayscale(rec.image_path) - img_mean) / img_std).astype(np.float32)
                h, w = image.shape
                gts = gt_points.get(rec.name, [])
                hm_t, off_t, omask_t = make_gt_targets(h, w, gts)

                image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
                hm_t = torch.from_numpy(hm_t).unsqueeze(0).float().to(device)
                off_t = torch.from_numpy(off_t).unsqueeze(0).float().to(device)
                omask_t = torch.from_numpy(omask_t).unsqueeze(0).float().to(device)

                outputs = model(image_t)
                loss, stats = candidate_loss_v2(
                    outputs, hm_t, off_t, omask_t, [], h, w,
                    w_hm=args.w_hm, w_qual=0.0, w_off=val_w_off, w_rank=0.0)
                val_loss += stats["loss_total"]
                nv += 1

        record = {
            "epoch": epoch,
            "train_loss": epoch_stats["loss_total"] / max(n, 1),
            "val_loss": val_loss / max(nv, 1),
            "loss_hm": epoch_stats["loss_hm"] / max(n, 1),
            "loss_qual": epoch_stats["loss_qual"] / max(n, 1),
            "loss_off": epoch_stats["loss_off"] / max(n, 1),
            "loss_rank": epoch_stats["loss_rank"] / max(n, 1),
            "stage": 1 if (args.stage1_epochs > 0 and epoch <= args.stage1_epochs) else 2,
        }
        history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            stage = record["stage"]
            print(f"Epoch {epoch} [S{stage}]: train={record['train_loss']:.4f} val={record['val_loss']:.4f} "
                  f"hm={record['loss_hm']:.4f} qual={record['loss_qual']:.4f} "
                  f"off={record['loss_off']:.4f} rank={record['loss_rank']:.4f}")

        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            save_checkpoint(out_dir / "best.pt", model, epoch=epoch, metrics=record)

    save_json(out_dir / "history.json", {"history": history})
    best = min(history, key=lambda r: r["val_loss"])
    save_json(out_dir / "best_metrics.json", best)
    print(f"\nBest: epoch {best['epoch']}, val_loss {best['val_loss']:.4f}")


if __name__ == "__main__":
    main()

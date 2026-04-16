#!/bin/bash
# V0 sweep: no-training baseline filters on IRSTD-1K
set -e

DATASET="irstd1k"
PT_CKPT="runs/irstd1k_point_detector/best.pt"
EL_CKPT="runs/irstd1k_cde_baseonly/best.pt"
OUTDIR="runs/v0_sweep"
mkdir -p "$OUTDIR"

echo "=== V0 Baseline Sweep ==="

# V0-baseline: no filters (reproduce current result)
echo "--- baseline (no filters) ---"
python eval_pipeline.py \
    --dataset-name $DATASET \
    --point-checkpoint $PT_CKPT \
    --ellipse-checkpoint $EL_CKPT \
    --output "$OUTDIR/baseline.json"

# V0-a: score threshold sweeps
for TH in 0.6 0.7 0.8 0.9; do
    echo "--- score_thresh=$TH ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --score-thresh $TH \
        --output "$OUTDIR/score_${TH}.json"
done

# V0-b: top-K sweeps
for K in 1 2 3 5 10; do
    echo "--- top_k=$K ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --top-k $K \
        --output "$OUTDIR/topk_${K}.json"
done

# V0-c: min distance suppression
for D in 5 8 10 15; do
    echo "--- min_distance_suppress=$D ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --min-distance-suppress $D \
        --output "$OUTDIR/suppress_${D}.json"
done

# V0-d: contrast filter
for C in 0.1 0.2 0.3 0.5; do
    echo "--- contrast_thresh=$C ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --contrast-thresh $C \
        --output "$OUTDIR/contrast_${C}.json"
done

# V0-e: ellipse area filter
for A in 100 200 400; do
    echo "--- max_ellipse_area=$A ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --max-ellipse-area $A \
        --output "$OUTDIR/maxarea_${A}.json"
done

# V0-f: center offset filter
for O in 2 3 4 5; do
    echo "--- max_center_offset=$O ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --max-center-offset $O \
        --output "$OUTDIR/maxoffset_${O}.json"
done

# V0-g: aspect ratio filter
for R in 3 5 8; do
    echo "--- max_aspect_ratio=$R ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --max-aspect-ratio $R \
        --output "$OUTDIR/maxaspect_${R}.json"
done

# V2: oracle verifier upper bounds
for R in 3 5 8; do
    echo "--- oracle_match_radius=$R ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --oracle-match-radius $R \
        --output "$OUTDIR/oracle_verifier_${R}.json"
done

echo "=== Done ==="

# Summary
echo ""
echo "=== SUMMARY ==="
python3 -c "
import json, glob
files = sorted(glob.glob('$OUTDIR/*.json'))
print(f'{\"config\":<30} {\"full_IoU\":>10} {\"full_nIoU\":>10} {\"inst_nIoU\":>10} {\"full_FA\":>12} {\"proposals\":>10} {\"post_filt\":>10}')
print('-' * 100)
for f in files:
    d = json.load(open(f))
    name = f.split('/')[-1].replace('.json','')
    print(f'{name:<30} {d[\"full_IoU\"]:10.4f} {d[\"full_nIoU\"]:10.4f} {d[\"instance_nIoU\"]:10.4f} {d[\"full_FA\"]:12.2e} {d[\"total_proposals\"]:10d} {d[\"total_post_filter\"]:10d}')
"

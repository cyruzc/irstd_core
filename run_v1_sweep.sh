#!/bin/bash
# V1 sweep: learned verifier with different thresholds
set -e

DATASET="irstd1k"
PT_CKPT="runs/irstd1k_point_detector/best.pt"
EL_CKPT="runs/irstd1k_cde_baseonly/best.pt"
VR_CKPT="runs/irstd1k_verifier/best.pt"
OUTDIR="runs/v1_sweep"

echo "=== V1 Verifier Sweep ==="

# Match the point threshold used in proposal generation
for VT in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo "--- verifier_thresh=$VT ---"
    python eval_pipeline.py \
        --dataset-name $DATASET \
        --point-checkpoint $PT_CKPT \
        --ellipse-checkpoint $EL_CKPT \
        --verifier-checkpoint $VR_CKPT \
        --verifier-threshold $VT \
        --point-threshold 0.3 \
        --output "$OUTDIR/verifier_${VT}.json"
done

echo "=== Summary ==="
python3 -c "
import json, glob
files = sorted(glob.glob('$OUTDIR/verifier_*.json'))
print(f'{\"config\":<25} {\"full_IoU\":>10} {\"full_nIoU\":>10} {\"inst_nIoU\":>10} {\"full_FA\":>12} {\"post_filt\":>10}')
print('-' * 80)
for f in files:
    d = json.load(open(f))
    name = f.split('/')[-1].replace('.json','')
    print(f'{name:<25} {d[\"full_IoU\"]:10.4f} {d[\"full_nIoU\"]:10.4f} {d[\"instance_nIoU\"]:10.4f} {d[\"full_FA\"]:12.2e} {d[\"total_post_filter\"]:10d}')
# Also print baselines
print()
print('--- Reference baselines ---')
for name, path in [('no_filter', 'runs/v0_sweep/baseline.json'), ('oracle_verif_3', 'runs/v0_sweep/oracle_verifier_3.json')]:
    if __import__('pathlib').Path(path).exists():
        d = json.load(open(path))
        print(f'{name:<25} {d[\"full_IoU\"]:10.4f} {d[\"full_nIoU\"]:10.4f} {d[\"instance_nIoU\"]:10.4f} {d[\"full_FA\"]:12.2e} {d[\"total_post_filter\"]:10d}')
"

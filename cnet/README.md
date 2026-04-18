# CNet: Center-Voting Candidate Network for Infrared Small Target Detection

A lightweight candidate proposal network for infrared small target detection, using center voting with peak suppression and redundancy elimination.

## Overview

CNet generates high-quality candidate regions through:
- **Center Scoring**: U-Net backbone predicts center probability maps
- **Offset Voting**: Each pixel votes for its nearest target center
- **Ring Suppression**: Penalize responses in annular regions around GT points
- **Redundancy Elimination**: Pairwise penalty for duplicate detections
- **Count Supervision**: Differentiable over/under-count constraints

## Architecture

```
Input (1-channel IR image)
    ↓
Encoder: Stem → Down1 → Down2 → Down3 (bottleneck)
    ↓
Decoder: Up2 → Up1 → Up0 (with skip connections)
    ↓
├─ Center Head → center_logits → sigmoid → center_prob
├─ Offset Head → (dy, dx) normalized offsets
└─ Vote Accumulation → proposal_map = vote_map × center_prob
```

### Loss Components

| Component | Weight | Description |
|-----------|--------|-------------|
| center_loss | 1.0 | Weighted BCE on Gaussian targets |
| dice_loss | 0.5 | Soft Dice loss |
| vote_loss | 1.0 | Supervision on voted proposals |
| ring_loss | 0.5 | Suppress responses in GT ring region |
| offset_loss | 1.0 | Smooth L1 on offset predictions |
| peak_ranking_loss | 0.2 | Margin-based peak dominance |
| count_loss | 0.05 | Over/under-count penalty |
| redundancy_loss | 0.2 | Pairwise proximity penalty |

## Installation

```bash
cd irstd_core
pip install -r requirements.txt
```

## Datasets

Supported datasets (auto-configured):

| Dataset | Mean | Std |
|---------|------|-----|
| SIRST3 | 95.01 | 41.51 |
| IRSTD-1K | 87.47 | 39.72 |
| NUAA-SIRST | 101.06 | 34.62 |
| NUDT-SIRST | 95.00 | 40.00 |

Dataset structure:
```
datasets/
└── IRSTD-1K/
    ├── images/
    ├── masks/
    ├── masks_centroid/      # point labels for training
    ├── masks_coarse/        # coarse labels (optional)
    └── img_idx/
        ├── train_IRSTD-1K.txt
        └── test_IRSTD-1K.txt
```

## Training

```bash
python train_candidate.py \
    --dataset-name irstd1k \
    --output-dir runs/cnet_irstd1k \
    --epochs 100 \
    --batch-size 16 \
    --lr 2.5e-4 \
    --base-channels 32 \
    --max-candidates 16 \
    --cache-data
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-name` | irstd1k | Dataset choice |
| `--base-channels` | 32 | Network width |
| `--max-candidates` | 16 | Max proposals per image |
| `--nms-kernel` | 7 | NMS window size |
| `--score-threshold` | 0.25 | Candidate score threshold |
| `--gaussian-sigma` | 2.0 | Center label spread |
| `--ring-inner-radius` | 2 | Inner suppression radius |
| `--ring-outer-radius` | 6 | Outer suppression radius |
| `--redundancy-radius` | 6.0 | Redundancy penalty scale |
| `--threshold-search-start` | 0.10 | F1 search range start |
| `--threshold-search-stop` | 0.80 | F1 search range stop |
| `--threshold-search-step` | 0.02 | F1 search step |

## Output

Training produces:
```
runs/cnet_irstd1k/
├── best.pt              # Best model checkpoint
├── history.json         # Training metrics history
└── (tensorboard logs)   # If using logger
```

## Citation

```bibtex

```

## License

MIT License

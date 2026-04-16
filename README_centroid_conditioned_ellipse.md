# Centroid-Conditioned Ellipse Reconstruction

This branch adds a validation framework for the hypothesis:

> Given an approximately correct target centroid, infrared small-target detection can be reformulated as a low-dimensional support-region reconstruction problem.

The current stage instantiates the support region as a **single ellipse**.

## Added files

- `ellipse_utils.py`: connected-component extraction, analytic ellipse fitting, crop/paste helpers, center hints.
- `ellipse_renderer.py`: differentiable soft ellipse renderer.
- `ellipse_losses.py`: region + moment + parameter losses.
- `ellipse_model.py`: centroid-conditioned patch-to-ellipse regressor.
- `ellipse_data.py`: instance-level centroid-conditioned dataset.
- `ellipse_engine.py`: training / instance-eval / full-image reconstruction eval.
- `train_ellipse.py`: end-to-end training entry.
- `eval_ellipse_upperbound.py`: analytic ellipse-fit upper bound.

## Training

```bash
python train_ellipse.py \
  --dataset-name sirst3 \
  --patch-size 32 \
  --epochs 100 \
  --batch-size 128 \
  --output-dir runs/sirst3_ellipse
```

## Evaluation only

```bash
python train_ellipse.py \
  --dataset-name sirst3 \
  --eval-only \
  --resume runs/sirst3_ellipse/best.pt \
  --output-dir runs/sirst3_ellipse_eval
```

## Analytic upper bound

```bash
python eval_ellipse_upperbound.py \
  --dataset-name sirst3 \
  --output runs/sirst3_ellipse_upperbound.json
```

## What is measured

### Instance level

- `instance_IoU`, `instance_nIoU`
- `instance_mae_dx`, `instance_mae_dy`
- `instance_mae_a`, `instance_mae_b`, `instance_mae_phi`

### Full image level

Each predicted local ellipse mask is pasted back to the image canvas using the known centroid-conditioned crop geometry. Final image-level metrics are:

- `full_IoU`
- `full_nIoU`
- `full_PD`
- `full_FA`

## Suggested protocol

1. Run `eval_ellipse_upperbound.py` to verify whether the **single-ellipse assumption** is sufficiently tight on each dataset.
2. Train `train_ellipse.py` with `--eval-prompt-noise-std 0.0` to test strict oracle-centroid reconstruction.
3. Increase `--eval-prompt-noise-std` to test robustness against centroid perturbation.
4. Compare against a local patch segmentation baseline if needed.

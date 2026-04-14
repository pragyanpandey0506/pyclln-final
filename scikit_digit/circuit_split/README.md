# `scikit_digit/circuit_split/`

Split-session scikit-digit hinge training for the no-hidden-node dense topology.

## Files

- `dense_trainer_split.py`
  - split-backend equivalent of `scikit_digit/dense_trainer.py`
  - creates 10 ngspice sessions, one per output branch
  - keeps the original free/clamp/update training order
- `compare_runs.py`
  - compares a split run against the saved best dense reference run
  - checks the saved history arrays plus per-epoch `vg`, confusion, and test-output dumps

## Best Dense Reference

The saved dense hinge reference selected for comparison is:

- `scikit_digit/results/dense_sweep_11jan2026/run_0489_seed1_g0.3_d0.05_m0.02_btfloating_vgfix4`

This is the best final-validation hinge run currently present under `scikit_digit/results/`.

## Example

Run the split trainer with the reference config:

```bash
conda run -n p311env python scikit_digit/circuit_split/dense_trainer_split.py \
  1 --epochs 100 --solver klu --gamma 0.3 --delta 0.05 --margin 0.02 \
  --body-tie floating --vg-init fixed --vg-init-fixed 4
```

Compare the split run against the saved dense reference:

```bash
conda run -n p311env python scikit_digit/circuit_split/compare_runs.py
```

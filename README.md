# pyclln-final

Analog NMOS network training on sklearn digits using ngspice via PySpice.

## Repo layout
- `device_model/` contains the NMOS model card and wrapper subckt.
- `scikit_digit/` contains training scripts, topology files, and run outputs.
- `scikit_digit/topology/` holds topology artifacts in `.npz` format.
- `scikit_digit/results/` is where runs are written (`runs/` + `latest`).
- `scikit_digit/results/auto_prune_sweep_11jan2026/` contains sweep runs.
- `scikit_digit/results/final_figures_22_jan/` contains the curated final-figures runs and plots (tracked in git).

## Requirements
- Python 3.x
- Python packages: `numpy`, `scikit-learn`, `networkx`, `matplotlib`, `PySpice`
- Ngspice shared library (`libngspice.so`) available on your system

## Quick start
Run dense training:
```
python3 scikit_digit/dense_trainer.py 0 --epochs 20
```

Run auto-prune training:
```
python3 scikit_digit/auto_prune_trainer.py 0 --epochs 20
```

Run the sweep launcher:
```
python3 scikit_digit/auto_prune_sweep.py
```

Run the cross-entropy sweep launcher:
```
python3 scikit_digit/xent_sweep.py
```

## Sweeps and recorded results
This repo keeps full sweep configurations in code, but only tracks the curated final-figures runs in git.
All other sweep outputs remain on disk under `scikit_digit/results/` and are excluded to keep the repo size sane.

### Hinge sweeps (dense + auto-prune)
Sweep launchers:
- `scikit_digit/dense_sweep.py` (hinge dense trainer)
- `scikit_digit/auto_prune_sweep.py` (hinge auto-prune trainer)

Sweep grid (both launchers use the same grid):
- epochs: 100
- gamma: 0.03, 0.3, 3.0
- delta: 0.02, 0.05, 0.1
- margin: 0.02, 0.05, 0.1
- body-tie: ground, source, floating
- vg-init:
  - fixed: 0.75, 2.0, 4.0
  - random: [0.75, 3.0], [1.0, 6.0], [1.5, 2.5]
- solver: klu
- seed: 0, 1, 2
- total runs per sweep: 3 (gamma) * 3 (delta) * 3 (margin) * 3 (body-tie) * 6 (vg-init) * 3 (seed) = 1458
- parallelism: dense sweep 10-way, auto-prune sweep 20-way

Outputs (per run directory):
- `run_meta.json`, `run_spec.json`, `train_log.txt`
- per-epoch summaries: `0_epoch_summary_epoch*.json`, `0_diag_epoch*.json`
- histories: `0_train_*.npy`, `0_val_*.npy`, timing arrays, reload/nonfinite arrays
- full test-set outputs: `0_vout_test_epoch*.npy`
- per-epoch confusion matrices (val): `0_val_confusion_epoch*.npy`
- plots: `learning_curves_*.png`, `timing.png`, `hinge_active.png` (auto-prune/dense)

### Cross-entropy sweep
Sweep launcher: `scikit_digit/xent_sweep.py`

Sweep grid:
- epochs: 20
- gamma: 0.03, 0.3, 3.0
- delta: 0.25, 0.5, 0.75
- softmax temperature: 0.0005, 0.001, 0.005, 0.01, 0.05
- body-tie: ground, source, floating
- vg-init:
  - fixed: 0.75, 2.0, 4.0
  - random: [0.75, 3.0], [1.0, 6.0], [1.5, 2.5]
- solver: klu
- seed: 0
- total runs: 3 (gamma) * 3 (delta) * 5 (temp) * 3 (body-tie) * 6 (vg-init) = 810
- parallelism: 10-way

Outputs (per run directory) match the hinge sweeps with CE-specific metrics (`0_train_ce.npy`, `0_val_ce.npy`).

### Final figures (22 Jan)
Tracked in git under `scikit_digit/results/final_figures_22_jan/`:
- `dense_best/` (hinge)
  - seed=1, epochs=10, gamma=0.3, delta=0.1, margin=0.02
  - body-tie=floating, vg-init=fixed (4.0), solver=klu
  - topology: `scikit_digit/topology/random_topology_0799.npz` (571 edges after pruning at epoch 64)
- `xent_best/` (cross entropy)
  - seed=0, epochs=10, gamma=0.3, delta=0.25, softmax-temp=0.001
  - body-tie=floating, vg-init=fixed (4.0), solver=klu
  - topology: `scikit_digit/topology/random_topology_0799.npz` (same pruned topology as above)

Final-figures folder contents:
- full run metadata + logs (`run_meta.json`, `train_log.txt`)
- per-epoch summaries + diagnostics
- per-epoch val confusion matrices (`0_val_confusion_epoch*.npy`)
- per-epoch test-set outputs (`0_vout_test_epoch*.npy`)
- per-epoch VG snapshots (`0_vg_unique_epoch*.npy`)
- plots (confusion matrices, VG heatmaps/histograms/graphs, accuracy/loss/hinge curves)

## Scripts
- `scikit_digit/dense_trainer.py`: hinge-loss trainer; loads topology from `.npz`.
- `scikit_digit/auto_prune_trainer.py`: hinge-loss trainer with edge pruning.
- `scikit_digit/auto_prune_sweep.py`: sweep launcher (writes under `scikit_digit/results/auto_prune_sweep_11jan2026`).
- `scikit_digit/dense_trainer_avgappr.py`: MSE-based trainer with prototype/averaging behavior.
- `scikit_digit/dense_trainer_cross_entropy.py`: cross-entropy trainer with softmax temperature.
- `scikit_digit/xent_sweep.py`: cross-entropy sweep launcher (writes under `scikit_digit/results/xent_sweep_11jan2026`).
- `scikit_digit/topology/topology_heatmap.ipynb`: visualizes per-output input connectivity heatmaps.

## Notes
- The trainers load topology from `.npz` under `scikit_digit/topology/`.
- `dense_trainer.py` and `auto_prune_trainer.py` support `--vg-init random|fixed` with `--vg-init-lo/--vg-init-hi` and `--vg-init-fixed`.
- `--body-tie` switches device body between source, ground, and floating; body resistor is fixed to `RS_CLAMP`.
- `auto_prune_trainer.py` computes `vg_cutoff = 0.8 * vto` from `device_model/nmos_lvl1_ald1106.lib` and prunes edges that stay below cutoff for the configured number of epochs (`--vg-cutoff-epochs`).
- `auto_prune_trainer.py` writes `topology_final_pruned.npz` into each run folder.
- Run artifacts are written under `scikit_digit/results/runs/` and `scikit_digit/results/latest` is updated to the most recent run.

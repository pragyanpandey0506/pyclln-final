# pyclln-final

Analog NMOS network training on sklearn digits using ngspice via PySpice.

## Repo layout
- `device_model/` contains the NMOS model card and wrapper subckt.
- `scikit_digit/` contains training scripts, topology files, and run outputs.
- `scikit_digit/topology/` holds topology artifacts in `.npz` format.
- `scikit_digit/results/` is where runs are written (`runs/` + `latest`).
- `scikit_digit/results/auto_prune_sweep_11jan2026/` contains sweep runs.

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

## Scripts
- `scikit_digit/dense_trainer.py`: hinge-loss trainer; loads topology from `.npz`.
- `scikit_digit/auto_prune_trainer.py`: hinge-loss trainer with edge pruning.
- `scikit_digit/auto_prune_sweep.py`: sweep launcher (writes under `scikit_digit/results/auto_prune_sweep_11jan2026`).
- `scikit_digit/dense_trainer_avgappr.py`: MSE-based trainer with prototype/averaging behavior.
- `scikit_digit/topology/topology_heatmap.ipynb`: visualizes per-output input connectivity heatmaps.

## Notes
- The trainers load topology from `.npz` under `scikit_digit/topology/`.
- `dense_trainer.py` and `auto_prune_trainer.py` support `--vg-init random|fixed` with `--vg-init-lo/--vg-init-hi` and `--vg-init-fixed`.
- `--body-tie` switches device body between source, ground, and floating; body resistor is fixed to `RS_CLAMP`.
- `auto_prune_trainer.py` computes `vg_cutoff = 0.8 * vto` from `device_model/nmos_lvl1_ald1106.lib` and prunes edges that stay below cutoff for the configured number of epochs (`--vg-cutoff-epochs`).
- `auto_prune_trainer.py` writes `topology_final_pruned.npz` into each run folder.
- Run artifacts are written under `scikit_digit/results/runs/` and `scikit_digit/results/latest` is updated to the most recent run.

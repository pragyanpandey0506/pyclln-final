# pyclln-final

Analog NMOS network training on sklearn digits using ngspice via PySpice.

## Repo layout
- `device_model/` contains the NMOS model card and wrapper subckt.
- `scikit_digit/` contains training scripts, topology files, and run outputs.
- `scikit_digit/topology/` holds topology artifacts in `.npz` format.
- `scikit_digit/results/` is where runs are written (`runs/` + `latest`).

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

## Notes
- The trainers load topology from `.npz` under `scikit_digit/topology/`.
- `auto_prune_trainer.py` computes `vg_cutoff` from `vto` in `device_model/nmos_lvl1_ald1106.lib` and prunes edges that stay below cutoff for the configured number of epochs (`--vg-cutoff-epochs`).
- Run artifacts are written under `scikit_digit/results/runs/` and `scikit_digit/results/latest` is updated to the most recent run.

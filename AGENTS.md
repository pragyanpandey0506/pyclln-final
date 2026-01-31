# AGENTS

Project-specific guidance for assistants working in this repo.

## Structure
- Training scripts live in `scikit_digit/` (`dense_trainer.py`, `auto_prune_trainer.py`, `dense_trainer_avgappr.py`, `dense_trainer_cross_entropy.py`).
- Sweep launchers are `scikit_digit/auto_prune_sweep.py` and `scikit_digit/xent_sweep.py`.
- Topology artifacts live in `scikit_digit/topology/` and are read via `topology_io.py`.
- Topology visualization lives in `scikit_digit/topology/topology_heatmap.ipynb`.
- Device model card is `device_model/nmos_lvl1_ald1106.lib`.
- Run outputs are written under `scikit_digit/results/runs/` with `scikit_digit/results/latest` pointing to the most recent run.
- Sweep outputs are written under `scikit_digit/results/auto_prune_sweep_11jan2026/`.
- Cross-entropy sweep outputs are written under `scikit_digit/results/xent_sweep_11jan2026/`.

## Practices
- Keep trainer logging and `run_meta.json` fields consistent when you add new options.
- If you change topology format or paths, update both the trainer config string and metadata.
- Avoid modifying the device model card unless explicitly requested.
- Run artifacts can be large; do not add or edit them unless the user asks.
- `auto_prune_trainer.py` saves `topology_final_pruned.npz` into each run folder.

# AGENTS

Project-specific guidance for assistants working in this repo.

## Structure
- Training scripts live in `scikit_digit/` (`dense_trainer.py`, `auto_prune_trainer.py`).
- Topology artifacts live in `scikit_digit/topology/` and are read via `topology_io.py`.
- Device model card is `device_model/nmos_lvl1_ald1106.lib`.
- Run outputs are written under `scikit_digit/results/runs/` with `scikit_digit/results/latest` pointing to the most recent run.

## Practices
- Keep trainer logging and `run_meta.json` fields consistent when you add new options.
- If you change topology format or paths, update both the trainer config string and metadata.
- Avoid modifying the device model card unless explicitly requested.
- Run artifacts can be large; do not add or edit them unless the user asks.

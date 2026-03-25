# AGENTS

Project-specific guidance for assistants working in this repo.

## Structure

- Digit training scripts live in `scikit_digit/`.
- Digit sweep launchers are `scikit_digit/auto_prune_sweep.py` and `scikit_digit/xent_sweep.py`.
- Digit topology artifacts live in `scikit_digit/topology/`.
- 16-token language-model scripts live in `lang_model/vocab16/`.
- 32-token language-model CE trainer lives in `lang_model/vocab32/clln_lang_ce_32_6.py`.
- Device model card is `device_model/nmos_lvl1_ald1106.lib`.

## Output Trees

- Digit run outputs live under `scikit_digit/results/`.
- 16-token language outputs live under:
  - `lang_model/vocab16/results_language_16_softce/`
  - `lang_model/vocab16/results_language_16_hinge/`
  - `lang_model/vocab16/results_language_16_linear_softce/`
  - `lang_model/vocab16/results_language_16_onehotce/`
- 32-token language outputs live under `lang_model/vocab32/results_language_32_softce/`.

These output trees can be large. Treat them as run artifacts, not source files.

## Practices

- Keep trainer logging and `run_meta.json` fields consistent when adding options.
- If you change topology format or result paths, update both the config string and metadata.
- Avoid modifying the device model card unless explicitly requested.
- Do not edit run artifacts unless the user asks.
- `auto_prune_trainer.py` saves `topology_final_pruned.npz` into each run folder.
- The language-model scripts use result directories relative to the script path unless `RUN_DIR` is set.

## Documentation

- Root overview: `README.md`
- Language-model top-level overview: `lang_model/README.md`
- 16-token language-model details: `lang_model/vocab16/README.md`
- 32-token language-model details: `lang_model/vocab32/README.md`

When repo structure changes, update both of those files.

# Language Model Experiments

`lang_model/` is now split by vocabulary size:

- `vocab16/`
  - all 16-token trainers, inference helpers, the BOS-padded quantum prompt-completion variant, baselines, and their result trees
- `vocab32/`
  - 32-token analog trainers, compatibility wrappers for moved linear assets, result trees, and analysis outputs
- `vocab32/linear_resistor/`
  - compatibility wrapper, sweep launcher, and notes for the moved vocab32 linear-resistor backend

Active linear work now lives outside this folder under:

- `../linear_net/lin_lang_model/`

Use this file for top-level orientation. Use the subfolder READMEs for task-specific details.

## Folder Layout

- `vocab16/`
  - 16-token / 6-token-context work
- `vocab32/`
  - 32-token / 6-token-context work
- `vocab32/linear_resistor/`
  - resistor-specific compatibility wrappers, sweep launcher, and notes for the moved embed4 one-hot CE linear backends
- `__pycache__/`
  - local Python cache files

## Main Entry Points

From repo root:

```bash
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce_quantum.py 0 --epochs 20 --warm-start-best-legacy
conda run -n p311env python lang_model/vocab16/clln_language_dense_trainer_16.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/linear_mlp_lang_ce.py 0 --epochs 20
conda run -n p311env python linear_net/lin_lang_model/vocab16/clln_lang_trainer_onehot_ce_linear_resistor.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/language16_infer_softce.py --prompt "the boy"
conda run -n p311env python lang_model/vocab16/language16_infer_quantum_onehot.py --prompt "the detector"
conda run -n p311env python lang_model/vocab16/analyze_best_onehot_support_model.py
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py 0 --epochs 35
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_remodel.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/run_gdt_init_sweep.py
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py 0 --epochs 20
```

## Result Trees

16-token outputs:

- `vocab16/results_language_16_softce/`
- `vocab16/results_language_16_onehotce/`
- `vocab16/results_language_16_onehotce_quantum/`
- `vocab16/results_language_16_hinge/`
- `vocab16/analysis_onehot_best_support_val/`
- `../linear_net/lin_lang_model/vocab16/results_language_16_linear_softce/`
- `../linear_net/lin_lang_model/vocab16/results_language_16_linear_resistor_onehotce/`

32-token outputs:

- `vocab32/results_language_32_softce/`
- `vocab32/results_language_32_embed4_onehotce/`
- `../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/`
- `../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_hiddenpath_onehotce/`
- `../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_remodel_onehotce/`
- `../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_onehotce/`

32-token analysis outputs:

- `vocab32/analysis_reports/`

## Current 32-Token Variants

- `clln_lang_ce_32_6.py`
  - 30-bit binary-token input, soft-target CE, analog NMOS trainer
- `clln_lang_trainer_embed4_onehot_ce.py`
  - 24-d fixed-embedding input, one-hot CE, analog NMOS trainer
- `clln_lang_trainer_embed4_onehot_ce_linear_resistor.py`
  - compatibility wrapper to the active 24-d fixed-embedding linear resistor KCL trainer under `../linear_net/lin_lang_model/vocab32/`
- `clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py`
  - compatibility wrapper to the active direct-plus-hidden-path linear resistor KCL trainer under `../linear_net/lin_lang_model/vocab32/`
- `clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py`
  - compatibility wrapper to the active linear resistor KCL crossbar trainer with slow edge birth/death remodeling under `../linear_net/lin_lang_model/vocab32/`
- `linear_resistor/`
  - compatibility entry points, exact 54-run `gamma/delta/temp/init` sweep launcher, and implementation-memory README for the moved resistor backends
- `linear_mlp_lang_embed4_onehot_ce.py`
  - compatibility wrapper to the active 24-d fixed-embedding linear no-hidden-layer baseline under `../linear_net/lin_lang_model/vocab32/`

## Notes

- The trainers still write results relative to their own script directory unless `RUN_DIR` is set.
- Large result trees are run artifacts, not source files.
- Paper-ready multi-panel figure bundles can live under `../figures/`.
- For repo-wide orientation, use `../README.md`.
- For the resistor-specific handoff notes and archived chat decisions, use `vocab32/linear_resistor/README.md`.
- For the active linear code and runs, use `../linear_net/lin_lang_model/`.

# Language Model Experiments

`lang_model/` is now split by vocabulary size:

- `vocab16/`
  - all 16-token trainers, inference helper, baselines, and their result trees
- `vocab32/`
  - 32-token trainers, the embed4 linear baseline, result trees, and analysis outputs

Use this file for top-level orientation. Use the subfolder READMEs for task-specific details.

## Folder Layout

- `vocab16/`
  - 16-token / 6-token-context work
- `vocab32/`
  - 32-token / 6-token-context work
- `__pycache__/`
  - local Python cache files

## Main Entry Points

From repo root:

```bash
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/clln_language_dense_trainer_16.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/linear_mlp_lang_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/language16_infer_softce.py --prompt "the boy"
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py 0 --epochs 35
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py 0 --epochs 20
```

## Result Trees

16-token outputs:

- `vocab16/results_language_16_softce/`
- `vocab16/results_language_16_onehotce/`
- `vocab16/results_language_16_hinge/`
- `vocab16/results_language_16_linear_softce/`

32-token outputs:

- `vocab32/results_language_32_softce/`
- `vocab32/results_language_32_embed4_onehotce/`
- `vocab32/results_language_32_embed4_linear_onehotce/`

32-token analysis outputs:

- `vocab32/analysis_reports/`

## Current 32-Token Variants

- `clln_lang_ce_32_6.py`
  - 30-bit binary-token input, soft-target CE, analog NMOS trainer
- `clln_lang_trainer_embed4_onehot_ce.py`
  - 24-d fixed-embedding input, one-hot CE, analog NMOS trainer
- `linear_mlp_lang_embed4_onehot_ce.py`
  - 24-d fixed-embedding input, one-hot CE, linear no-hidden-layer baseline

## Notes

- The trainers still write results relative to their own script directory unless `RUN_DIR` is set.
- Large result trees are run artifacts, not source files.
- For repo-wide orientation, use `../README.md`.

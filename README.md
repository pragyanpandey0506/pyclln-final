# pyclln-final

Analog NMOS network training with ngspice via PySpice.

The repo now has two main work areas:

- `scikit_digit/` for the original sklearn-digit experiments
- `lang_model/` for the synthetic language-model experiments

The language-model area is now organized by vocabulary size:

- `lang_model/vocab16/`
- `lang_model/vocab32/`

## Layout

- `device_model/`
  - NMOS model card and wrapper subcircuit (`nmos_lvl1_ald1106.lib`)
- `scikit_digit/`
  - digit trainers, sweep launchers, topology files, and curated result folders
- `lang_model/`
  - split into `vocab16/` and `vocab32/` attempts, with local docs and result trees

## Environment

Expected environment is `p311env` with:

- Python 3.11
- `numpy`, `scikit-learn`, `networkx`, `matplotlib`
- `PySpice`
- system `ngspice` shared library (`libngspice.so`)

Examples:

```bash
conda run -n p311env python scikit_digit/dense_trainer.py --help
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py --help
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py --help
```

## Quick Start

Digit trainer:

```bash
conda run -n p311env python scikit_digit/dense_trainer.py 0 --epochs 20
```

16-token language CE trainer:

```bash
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py 0 --epochs 20
```

16-token language hinge trainer:

```bash
conda run -n p311env python lang_model/vocab16/clln_language_dense_trainer_16.py 0 --epochs 20
```

32-token language CE trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py 0 --epochs 35
```

32-token embed4 one-hot trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0 --epochs 20
```

32-token embed4 linear one-hot baseline:

```bash
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py 0 --epochs 20
```

## Language-Model Work

### `lang_model/`

`lang_model/` now contains two language-model areas:

- `lang_model/vocab16/`
  - 16-token trainers, inference helper, baseline, and 16-token result trees
- `lang_model/vocab32/`
  - 32-token trainers, linear baseline, analysis outputs, and 32-token result trees

Run outputs for these scripts are written under:

- `lang_model/vocab16/results_language_16_softce/`
- `lang_model/vocab16/results_language_16_hinge/`
- `lang_model/vocab16/results_language_16_linear_softce/`
- `lang_model/vocab16/results_language_16_onehotce/`
- `lang_model/vocab32/results_language_32_softce/`
- `lang_model/vocab32/results_language_32_embed4_onehotce/`
- `lang_model/vocab32/results_language_32_embed4_linear_onehotce/`

Derived sweep summaries and analysis reports for the 32-token work live under:

- `lang_model/vocab32/analysis_reports/`

Detailed documentation for the 16-token setup lives in:

- `lang_model/vocab16/README.md`

Detailed documentation for the 32-token setup lives in:

- `lang_model/vocab32/README.md`

## Digit Work

Main digit scripts:

- `scikit_digit/dense_trainer.py`
- `scikit_digit/auto_prune_trainer.py`
- `scikit_digit/dense_trainer_avgappr.py`
- `scikit_digit/dense_trainer_cross_entropy.py`
- `scikit_digit/auto_prune_sweep.py`
- `scikit_digit/xent_sweep.py`

Topology artifacts:

- `scikit_digit/topology/`
- `scikit_digit/topology/topology_heatmap.ipynb`

Tracked curated results:

- `scikit_digit/results/final_figures_22_jan/`

## Notes

- The analog trainers typically write results relative to the script directory unless `RUN_DIR` is provided.
- Large run-output trees are intentionally ignored in git.
- The top-level language-model layout is documented in `lang_model/README.md`.

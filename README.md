# pyclln-final

Analog NMOS network training with ngspice via PySpice.

The repo now has four main work areas:

- `scikit_digit/` for the original sklearn-digit experiments
- `lang_model/` for the synthetic language-model experiments
- `linear_net/` for the reorganized linear and linear-resistor language-model work
- `nonlin_reg/` for the archived-style 4x4 non-linear regression starter
- `figures/` for paper-ready figure bundles and plotting scripts

The language-model area is now organized by vocabulary size:

- `lang_model/vocab16/`
- `lang_model/vocab32/`

The active linear language-model work now lives under:

- `linear_net/lin_lang_model/`

Older linear paths under `lang_model/` are kept as compatibility wrappers.

## Layout

- `device_model/`
  - NMOS model card and wrapper subcircuit (`nmos_lvl1_ald1106.lib`)
- `scikit_digit/`
  - digit trainers, sweep launchers, topology files, split-backend experiments under `circuit_split/`, and curated result folders
- `lang_model/`
  - split into `vocab16/` and `vocab32/` attempts, with local docs and result trees
- `linear_net/`
  - reorganized home for linear baselines, linear-resistor trainers, launchers, and their result trees
- `nonlin_reg/`
  - 4x4 non-linear regression starter aligned to the archived small-grid PNAS-style reproduction
- `figures/`
  - organized figure bundles, each with its own plotting script and exported assets

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
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce_quantum.py --help
conda run -n p311env python lang_model/vocab16/analyze_best_onehot_support_model.py --help
conda run -n p311env python linear_net/lin_lang_model/vocab16/clln_lang_trainer_onehot_ce_linear_resistor.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py --help
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py --help
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce.py --help
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path.py --help
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_remodel.py --help
conda run -n p311env python lang_model/vocab32/linear_resistor/run_gdt_init_sweep.py --help
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py --help
```

## Quick Start

Digit trainer:

```bash
conda run -n p311env python scikit_digit/dense_trainer.py 0 --epochs 20
```

4x4 non-linear regression starter:

```bash
conda run -n p311env python nonlin_reg/nonlin_reg_4x4_baseline.py --epochs 20
```

16-token language CE trainer:

```bash
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py 0 --epochs 20
```

16-token language hinge trainer:

```bash
conda run -n p311env python lang_model/vocab16/clln_language_dense_trainer_16.py 0 --epochs 20
```

16-token linear-resistor one-hot trainer:

```bash
conda run -n p311env python linear_net/lin_lang_model/vocab16/clln_lang_trainer_onehot_ce_linear_resistor.py 0 --epochs 20
```

32-token language CE trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py 0 --epochs 35
```

32-token embed4 one-hot trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0 --epochs 20
```

32-token embed4 linear-resistor one-hot trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py 0 --epochs 20
```

32-token embed4 linear-resistor folder wrapper:

```bash
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce.py 0 --epochs 20
```

32-token embed4 linear-resistor hidden-path trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py 0 --epochs 20
```

32-token embed4 linear-resistor crossbar remodeling trainer:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py 0 --epochs 20
```

32-token embed4 linear one-hot baseline:

```bash
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py 0 --epochs 20
```

## Language-Model Work

### `lang_model/`

`lang_model/` now contains two language-model areas:

- `lang_model/vocab16/`
  - 16-token trainers, inference helpers, the BOS-padded quantum prompt-completion variant, baseline, and 16-token result trees
- `lang_model/vocab32/`
  - 32-token analog trainers, analysis outputs, compatibility wrappers for moved linear assets, and analog result trees
- `lang_model/vocab32/linear_resistor/`
  - compatibility wrapper/launcher/docs area for moved vocab32 linear-resistor assets
- `linear_net/lin_lang_model/`
  - active linear and linear-resistor language-model code and result trees

Run outputs for these scripts are written under:

- `lang_model/vocab16/results_language_16_softce/`
- `lang_model/vocab16/results_language_16_hinge/`
- `lang_model/vocab16/results_language_16_linear_softce/`
- `lang_model/vocab16/results_language_16_onehotce/`
- `lang_model/vocab16/results_language_16_onehotce_quantum/`
- `lang_model/vocab16/analysis_onehot_best_support_val/`
- `lang_model/vocab32/results_language_32_softce/`
- `lang_model/vocab32/results_language_32_embed4_onehotce/`
- `linear_net/lin_lang_model/vocab16/results_language_16_linear_softce/`
- `linear_net/lin_lang_model/vocab16/results_language_16_linear_resistor_onehotce/`
- `linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/`
- `linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_hiddenpath_onehotce/`
- `linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_remodel_onehotce/`
- `linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_onehotce/`

Derived sweep summaries and analysis reports for the 32-token work live under:

- `lang_model/vocab32/analysis_reports/`

Detailed documentation for the 16-token setup lives in:

- `lang_model/vocab16/README.md`

Detailed documentation for the 32-token setup lives in:

- `lang_model/vocab32/README.md`
- `lang_model/vocab32/linear_resistor/README.md`

Linear-area documentation lives in:

- `linear_net/README.md`
- `linear_net/lin_lang_model/README.md`

## Digit Work

Main digit scripts:

- `scikit_digit/dense_trainer.py`
- `scikit_digit/auto_prune_trainer.py`
- `scikit_digit/dense_trainer_avgappr.py`
- `scikit_digit/dense_trainer_cross_entropy.py`
- `scikit_digit/circuit_split/dense_trainer_split.py`
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

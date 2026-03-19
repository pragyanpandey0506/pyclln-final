# pyclln-final

Analog NMOS network training with ngspice via PySpice.

The repo now has two main work areas:

- `scikit_digit/` for the original sklearn-digit experiments
- `lang_model/` plus `clln_lang_ce_32_6.py` for the synthetic language-model experiments

## Layout

- `device_model/`
  - NMOS model card and wrapper subcircuit (`nmos_lvl1_ald1106.lib`)
- `scikit_digit/`
  - digit trainers, sweep launchers, topology files, and curated result folders
- `lang_model/`
  - 16-token language trainers, inference helper, linear baseline, and docs
- `clln_lang_ce_32_6.py`
  - 32-token / 6-token-context soft cross-entropy language trainer
- `results_language_32_softce/`
  - root-level output tree for `clln_lang_ce_32_6.py`

## Environment

Expected environment is `p311env` with:

- Python 3.11
- `numpy`, `scikit-learn`, `networkx`, `matplotlib`
- `PySpice`
- system `ngspice` shared library (`libngspice.so`)

Examples:

```bash
conda run -n p311env python scikit_digit/dense_trainer.py --help
conda run -n p311env python lang_model/clln_lang_trainer_ce.py --help
conda run -n p311env python clln_lang_ce_32_6.py --help
```

## Quick Start

Digit trainer:

```bash
conda run -n p311env python scikit_digit/dense_trainer.py 0 --epochs 20
```

16-token language CE trainer:

```bash
conda run -n p311env python lang_model/clln_lang_trainer_ce.py 0 --epochs 20
```

16-token language hinge trainer:

```bash
conda run -n p311env python lang_model/clln_language_dense_trainer_16.py 0 --epochs 20
```

32-token language CE trainer:

```bash
conda run -n p311env python clln_lang_ce_32_6.py 0 --epochs 35
```

## Language-Model Work

### `lang_model/`

`lang_model/` contains the moved 16-token language experiments:

- `clln_lang_trainer_ce.py`
  - dense 24 -> 16 soft-target cross-entropy analog trainer
- `clln_language_dense_trainer_16.py`
  - dense 24 -> 16 hinge-set analog trainer
- `language16_infer_softce.py`
  - autoregressive inference helper for the best 16-token CE checkpoint
- `linear_mlp_lang_ce.py`
  - linear 24 -> 16 softmax baseline for comparison against the analog CE trainer

Run outputs for these scripts are written under:

- `lang_model/results_language_16_softce/`
- `lang_model/results_language_16_hinge/`
- `lang_model/results_language_16_linear_softce/`

Detailed documentation for the 16-token setup lives in:

- `lang_model/README.md`

### `clln_lang_ce_32_6.py`

This is the larger language-model experiment at repo root:

- context length: 6 tokens
- vocabulary size: 32 tokens
- soft-target cross-entropy clamp rule
- outputs written under `results_language_32_softce/`

That folder is run-artifact storage only:

- `results_language_32_softce/runs/`
- `results_language_32_softce/sweeps/`

The tracked repo does not keep those run artifacts by default.

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
- The most current documentation for language-model experiment organization is in `lang_model/README.md`.

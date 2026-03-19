# Language Model Experiments

This folder contains the 16-token synthetic language-model work and its local utilities.

It does not contain the larger 32-token CE trainer itself. That script stays at repo root as:

- `../clln_lang_ce_32_6.py`

## In This Folder

- `clln_lang_trainer_ce.py`
  - dense 24 -> 16 analog soft-target cross-entropy trainer
- `clln_language_dense_trainer_16.py`
  - dense 24 -> 16 analog hinge-set trainer
- `linear_mlp_lang_ce.py`
  - linear 24 -> 16 softmax baseline using the same input/output structure as the analog CE trainer
- `language16_infer_softce.py`
  - inference helper for the best 16-token CE checkpoint

## Result Trees

Tracked code lives here. Large run artifacts are written outside git by default:

- `results_language_16_softce/`
- `results_language_16_hinge/`
- `results_language_16_linear_softce/`

The related 32-token experiment writes to:

- `../results_language_32_softce/`

## 16-Token Task

Shared task structure for the analog CE trainer, analog hinge trainer, and linear baseline:

- vocabulary size: `16`
- context length: `6`
- token encoding: `4` bits per token
- input width: `24`
- output width: `16`
- dataset: synthetic grammar-generated text, created inside the trainer

Common dataset knobs:

- `--num-sentences`
- `--val-frac`
- `--template-mode`
- `--bit-v0`, `--bit-v1`

## Running

From repo root:

```bash
conda run -n p311env python lang_model/clln_lang_trainer_ce.py 0 --epochs 20
conda run -n p311env python lang_model/clln_language_dense_trainer_16.py 0 --epochs 20
conda run -n p311env python lang_model/linear_mlp_lang_ce.py 0 --epochs 20
```

Inference on the best saved 16-token CE checkpoint:

```bash
conda run -n p311env python lang_model/language16_infer_softce.py --prompt "the boy"
```

To force any trainer into a specific output folder:

```bash
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/clln_lang_trainer_ce.py 0
```

## Output Files

Typical run outputs include:

- `run_meta.json`
- `train_log.txt`
- `netlist_initial.cir`
- `vocab.json`
- `sample_sentences.txt`
- `context_target_preview.json`
- `samples_epoch*.txt`
- `0_epoch_summary_epoch*.json`
- `0_diag_epoch*.json`
- metric histories in `.npy`
- plots such as `learning_curves_*.png` and `timing.png`

Additional model-specific outputs:

- analog trainers:
  - `0_vg_unique_epoch*.npy`
- linear baseline:
  - `linear_model_final.npz`
  - `summary.json`

## Current 16-Token Artifacts

Known local runs and sweeps referenced during development:

- analog CE single run:
  - `results_language_16_softce/runs/20260317-163957_seed-0`
- analog CE sweep:
  - `results_language_16_softce/sweeps/gdt_20260317-165650`
- analog hinge single run:
  - `results_language_16_hinge/runs/20260317-155730_seed-0`
- linear baseline run:
  - `results_language_16_linear_softce/runs/20260318-132805_seed-0`

Best 16-token analog CE sweep results observed:

- best `support_acc`: `0.995471` at `g0.1_d0.1_t0.01`, epoch `11`
- best `exact_acc`: `0.538949` at `g0.03_d0.1_t0.01`, epoch `20`
- best `qmass_mean`: `0.781584` at `g0.1_d0.1_t0.01`, epoch `20`

## 32-Token CE Work

The broader language experiment is not inside this folder, but it is part of the same project:

- script: `../clln_lang_ce_32_6.py`
- output tree: `../results_language_32_softce/`

Recent interrupted-sweep summaries generated during analysis are typically written to:

- `analysis_reports/`

These are analysis outputs, not canonical source documentation.

## Notes

- The analog CE and hinge trainers write results relative to their own script directory.
- Older `scikit_digit/...` references for these 16-token trainers are stale; use `lang_model/...`.
- For repo-wide orientation, use `../README.md`.

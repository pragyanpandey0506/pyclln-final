# 16-Token Language Attempts

This folder contains the 16-token synthetic language-model work and its local utilities.

## In This Folder

- `clln_lang_trainer_ce.py`
  - dense 24 -> 16 analog soft-target cross-entropy trainer
- `clln_lang_trainer_onehot_ce.py`
  - dense 24 -> 16 analog one-hot cross-entropy trainer
- `clln_language_dense_trainer_16.py`
  - dense 24 -> 16 analog hinge-set trainer
- `linear_mlp_lang_ce.py`
  - linear 24 -> 16 softmax baseline using the same input/output structure as the analog CE trainer
- `language16_infer_softce.py`
  - inference helper for the best 16-token CE checkpoint

## Result Trees

- `results_language_16_softce/`
- `results_language_16_onehotce/`
- `results_language_16_hinge/`
- `results_language_16_linear_softce/`

## 16-Token Task

Shared task structure for the analog CE trainer, analog one-hot trainer, analog hinge trainer, and linear baseline:

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
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/clln_language_dense_trainer_16.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/linear_mlp_lang_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/language16_infer_softce.py --prompt "the boy"
```

To force any trainer into a specific output folder:

```bash
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab16/clln_lang_trainer_ce.py 0
```

## Output Files

Typical run outputs include:

- `run_meta.json`
- `train_log.txt`
- `netlist_initial.cir`
- `vocab.json`
- `sample_sentences.txt`
- `sample_target_preview.json`
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

## Notes

- The analog CE, one-hot CE, and hinge trainers write results relative to their own script directory.
- The linear baseline writes to `results_language_16_linear_softce/` in this folder unless `RUN_DIR` is set.
- Older references to `lang_model/...` without `vocab16/` are stale.
- For top-level language-model orientation, use `../README.md`.

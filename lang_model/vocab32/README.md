# 32-Token Language Attempts

This folder contains the larger 32-token synthetic language-model work.

It currently has two distinct 32-token setups:

- binary-token soft-CE analog training (`clln_lang_ce_32_6.py`)
- fixed-embedding one-hot training, with both analog and linear baselines

## In This Folder

- `clln_lang_ce_32_6.py`
  - dense 30 -> 32 analog soft-target cross-entropy trainer
- `clln_lang_trainer_embed4_onehot_ce.py`
  - dense 24 -> 32 analog one-hot cross-entropy trainer using fixed 4D token embeddings
- `linear_mlp_lang_embed4_onehot_ce.py`
  - linear 24 -> 32 softmax baseline trained with one-hot cross-entropy on the same embed4 task
- `results_language_32_softce/`
  - run outputs for the 32-token trainer
- `results_language_32_embed4_onehotce/`
  - run outputs for the 32-token embed4 analog one-hot trainer
- `results_language_32_embed4_linear_onehotce/`
  - run outputs for the 32-token embed4 linear one-hot baseline
- `analysis_reports/`
  - analysis outputs generated during sweeps and checkpoint inspection

## 32-Token Task

- `clln_lang_ce_32_6.py`
  - vocabulary size: `32`
  - context length: `6`
  - token encoding: `5` bits per token
  - input width: `30`
  - output width: `32`
- `clln_lang_trainer_embed4_onehot_ce.py` and `linear_mlp_lang_embed4_onehot_ce.py`
  - input vocabulary size: `33` including input-only `<BOS>`
  - output vocabulary size: `32`
  - context length: `6`
  - token encoding: fixed `4D` embedding per token
  - input width: `24`
  - output width: `32`

## Running

From repo root:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_ce_32_6.py 0 --epochs 35
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py 0 --epochs 20
```

To force outputs into a specific folder:

```bash
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0
```

## Result Conventions

- `results_language_32_softce/`
  - binary-token analog soft-CE runs
- `results_language_32_embed4_onehotce/`
  - fixed-embedding analog one-hot runs
- `results_language_32_embed4_linear_onehotce/`
  - fixed-embedding linear baseline runs
- `analysis_reports/`
  - derived summaries, sweep tables, and inspection notes

Typical run outputs include:

- `run_meta.json`
- `sample_sentences.txt`
- `0_epoch_summary_epoch*.json`
- metric histories in `.npy`
- `train_log.txt` for analog runs
- `linear_model_final.npz` and `summary.json` for the linear baseline

## Metrics

- `exact_acc`
  - argmax prediction matches the observed next token
- `support_acc`
  - argmax prediction falls in the support of the empirical train-split target distribution for that context
- `soft_ce`
  - CE against either the train-split empirical target distribution or the one-hot label, depending on trainer
- `qmass_mean`
  - predicted probability mass assigned to the target support

## Notes

- The trainer writes outputs relative to this script directory unless `RUN_DIR` is set.
- `analysis_reports/` contains derived summaries, not canonical source code.

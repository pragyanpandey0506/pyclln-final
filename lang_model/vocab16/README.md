# 16-Token Language Attempts

This folder contains the 16-token synthetic language-model work and its local utilities.

The active linear assets for this task now live under:

- `../../linear_net/lin_lang_model/vocab16/`

The `linear_mlp_lang_ce.py` file kept here is a compatibility wrapper to that moved baseline.

## In This Folder

- `clln_lang_trainer_ce.py`
  - dense 24 -> 16 analog soft-target cross-entropy trainer
- `clln_lang_trainer_onehot_ce.py`
  - dense 24 -> 16 analog one-hot cross-entropy trainer
- `clln_lang_trainer_onehot_ce_quantum.py`
  - dense 24 -> 15 analog one-hot trainer with input-only BOS padding, quantum-themed vocabulary, grammar-support metrics, optional forbidden-choice deterrence, and optional warm start from the best legacy one-hot run
- `clln_language_dense_trainer_16.py`
  - dense 24 -> 16 analog hinge-set trainer
- `linear_mlp_lang_ce.py`
  - compatibility wrapper to the moved linear 24 -> 16 softmax baseline
- `../../linear_net/lin_lang_model/vocab16/clln_lang_trainer_onehot_ce_linear_resistor.py`
  - active dense 24 -> 16 linear-resistor one-hot cross-entropy trainer using the same dataset and CLI shape as the analog one-hot trainer
- `language16_infer_softce.py`
  - inference helper for the best 16-token CE checkpoint
- `language16_infer_quantum_onehot.py`
  - prompt-conditioned inference helper for the quantum 15-output one-hot trainer
- `analyze_best_onehot_support_model.py`
  - selects the best one-hot CE checkpoint among `support_acc = 1.0` runs, then writes LM-style diagnostics and figures

## Result Trees

- `results_language_16_softce/`
- `results_language_16_onehotce/`
- `results_language_16_onehotce_quantum/`
- `results_language_16_hinge/`
- `../../linear_net/lin_lang_model/vocab16/results_language_16_linear_softce/`
- `../../linear_net/lin_lang_model/vocab16/results_language_16_linear_resistor_onehotce/`

## Analysis Outputs

- `analysis_onehot_best_support_val/`
  - generated figures, tables, and report for the best 16-token one-hot CE support-perfect checkpoint

## 16-Token Task

Shared task structure for the analog CE trainer, analog one-hot trainer, analog hinge trainer, linear baseline, and linear-resistor one-hot trainer:

- vocabulary size: `16`
- context length: `6`
- token encoding: `4` bits per token
- input width: `24`
- output width: `16`
- dataset: synthetic grammar-generated text, created inside the trainer

Quantum one-hot variant:

- input vocabulary size: `16`, with `<BOS>` used only as padding in the 6-token context window
- output vocabulary size: `15`, with `<BOS>` removed from the predicted token set
- domain: small quantum-themed prompt completion with grammar and semantic-compatibility rules
- extra validation metrics: grammar `support_acc`, `valid_mass`, `valid_logprob`, and `forbidden_mass`

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
conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce_quantum.py 0 --epochs 20 --warm-start-best-legacy
conda run -n p311env python lang_model/vocab16/clln_language_dense_trainer_16.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/linear_mlp_lang_ce.py 0 --epochs 20
conda run -n p311env python linear_net/lin_lang_model/vocab16/clln_lang_trainer_onehot_ce_linear_resistor.py 0 --epochs 20
conda run -n p311env python lang_model/vocab16/language16_infer_softce.py --prompt "the boy"
conda run -n p311env python lang_model/vocab16/language16_infer_quantum_onehot.py --prompt "the detector"
conda run -n p311env python lang_model/vocab16/analyze_best_onehot_support_model.py
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
  - the quantum one-hot trainer also writes grammar-support histories such as `0_val_support_acc.npy`, `0_val_valid_mass.npy`, and `0_val_forbidden_mass.npy`
- linear baseline:
  - `linear_model_final.npz`
  - `summary.json`
- linear-resistor one-hot trainer:
  - `0_vg_unique_epoch*.npy`
  - `0_epoch_summary_epoch*.json`
  - `0_val_qmass.npy`

## Notes

- The analog CE, one-hot CE, and hinge trainers write results relative to their own script directory.
- The quantum one-hot trainer writes to `results_language_16_onehotce_quantum/` and expects prompt-conditioned generation rather than BOS prediction.
- The active linear baseline and linear-resistor trainer write to `../../linear_net/lin_lang_model/vocab16/` unless `RUN_DIR` is set.
- Older references to `lang_model/...` without `vocab16/` are stale.
- For top-level language-model orientation, use `../README.md`.

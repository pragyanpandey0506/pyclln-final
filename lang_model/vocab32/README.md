# 32-Token Language Attempts

This folder contains the larger 32-token synthetic language-model work.

It currently has two distinct 32-token setups:

- binary-token soft-CE analog training (`clln_lang_ce_32_6.py`)
- fixed-embedding one-hot training, with analog plus compatibility wrappers for the moved linear and linear-resistor variants
- the active linear assets now live under `../../linear_net/lin_lang_model/vocab32/`

## In This Folder

- `clln_lang_ce_32_6.py`
  - dense 30 -> 32 analog soft-target cross-entropy trainer
- `clln_lang_trainer_embed4_onehot_ce.py`
  - dense 24 -> 32 analog one-hot cross-entropy trainer using fixed 4D token embeddings
- `clln_lang_trainer_embed4_onehot_ce_linear_resistor.py`
  - compatibility wrapper to the moved dense 24 -> 32 linear-resistor one-hot cross-entropy trainer
- `clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py`
  - compatibility wrapper to the moved dense 24 -> 32 linear-resistor one-hot cross-entropy trainer with both direct and per-pair hidden paths
- `clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py`
  - compatibility wrapper to the moved dense 24 -> 32 linear-resistor one-hot cross-entropy trainer with slow edge birth/death remodeling on the direct crossbar
- `linear_resistor/`
  - compatibility launchers, hard-example fine-tune launcher, exact 54-run sweep launcher, and a README that records the design and run context for the moved linear-resistor backends
- `linear_mlp_lang_embed4_onehot_ce.py`
  - compatibility wrapper to the moved linear 24 -> 32 softmax baseline
- `results_language_32_softce/`
  - run outputs for the 32-token trainer
- `results_language_32_embed4_onehotce/`
  - run outputs for the 32-token embed4 analog one-hot trainer
- `analysis_reports/`
  - analysis outputs generated during sweeps and checkpoint inspection
- `../../linear_net/lin_lang_model/vocab32/`
  - active linear-resistor and linear baseline trainers plus their result trees

## 32-Token Task

- `clln_lang_ce_32_6.py`
  - vocabulary size: `32`
  - context length: `6`
  - token encoding: `5` bits per token
  - input width: `30`
  - output width: `32`
- `clln_lang_trainer_embed4_onehot_ce.py`, `clln_lang_trainer_embed4_onehot_ce_linear_resistor.py`, `clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py`, and `linear_mlp_lang_embed4_onehot_ce.py`
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
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path_hard_examples.py --epochs 25
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_remodel.py 0 --epochs 20
conda run -n p311env python lang_model/vocab32/linear_resistor/run_gdt_init_sweep.py
conda run -n p311env python lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py 0 --epochs 20
```

To force outputs into a specific folder:

```bash
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py 0
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py 0
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce.py 0
RUN_DIR=/abs/path/to/run_dir conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py 0
```

## Result Conventions

- `results_language_32_softce/`
  - binary-token analog soft-CE runs
- `results_language_32_embed4_onehotce/`
  - fixed-embedding analog one-hot runs
- `../../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/`
  - fixed-embedding linear-resistor one-hot runs
- `../../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_hiddenpath_onehotce/`
  - fixed-embedding linear-resistor one-hot runs with direct plus hidden paths
- `../../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_resistor_remodel_onehotce/`
  - fixed-embedding linear-resistor one-hot runs with slow edge birth/death remodeling on the direct crossbar
- `../../linear_net/lin_lang_model/vocab32/results_language_32_embed4_linear_onehotce/`
  - fixed-embedding linear baseline runs
- `analysis_reports/`
  - derived summaries, sweep tables, and inspection notes
- `linear_resistor/README.md`
  - implementation notes, equations, sweep parameters, best support result noted in chat, and logging changes for the resistor backends

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
- The linear and linear-resistor paths in this folder are compatibility wrappers. The active code and result trees now live under `../../linear_net/lin_lang_model/vocab32/`.
- `train_embed4_onehot_ce_hidden_path_hard_examples.py` is a resume-style launcher that starts from the best saved hidden-path checkpoint, mines hard training rows from that checkpoint, mixes in replay examples, and then runs a new fine-tune job in a fresh output folder.
- The remodeling crossbar variant currently implements slow edge birth/death only. It does not implement node split/merge because the direct crossbar has no hidden-node pool.

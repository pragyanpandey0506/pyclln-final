# Vocab32 Linear Resistor

This folder is the project-memory and entry-point area for the 32-token embed4 linear-resistor implementation.

The actual trainer code remains at:

- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py`

That path was kept stable so existing commands, run logs, and result trees do not break. This folder adds:

- a wrapper entry point
- a hidden-path wrapper entry point
- a hidden-path hard-example fine-tune launcher
- a crossbar-remodeling wrapper entry point
- the exact 5-worker sweep launcher used in this chat
- a written handoff of the implementation decisions and run context

## Why This Exists

The original discussion started with a question about whether the CLLN scripts could be rewritten to solve with linear resistors using direct KCL/KVL instead of SPICE. The first assessment was:

- yes, the backend can be swapped because the trainers are structured around free-phase and clamp-phase node voltages, followed by a local edge update
- a resistor-only replacement is most straightforward when the circuit is linear and the topology is easy to solve by nodal analysis
- for dense input/output language setups, the resistor version is fully linear because there are no hidden nodes

The first proposed target was `nonlin_reg`, because that topology is graph-shaped and has internal nodes. That was then explicitly redirected to the 32-token vocab script, one-hot CE, with the request to run it exactly like the current analog trainer but with linear resistors.

## What Was Implemented

A separate trainer was created:

- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py`

It keeps the same task and operating style as the analog embed4 one-hot CE trainer:

- same 32-token synthetic language task
- same fixed 4D token embeddings
- same one-hot CE clamp construction
- same fresh-process `controller -> worker epoch` execution flow
- same broad CLI shape
- same `RUN_DIR` behavior

What changed is the circuit backend:

- ngspice was removed from the solve path
- each trainable edge is treated as a linear resistor parameter mapped to conductance
- output voltages are solved analytically with KCL

The resistor backend is dense `24 -> 32`, with one trainable resistor from every input feature to every output node.

## Linear KCL Model

For each output node `k`, the free-phase solve is:

```text
V_out[k] = (sum_i g[k,i] * x[i]) / (sum_i g[k,i] + 1 / RS_FREE)
```

For the clamp phase, the target voltage is:

```text
V_clamp_target = V_free + delta * (q - softmax(V_free / T))
```

and the clamped solve is:

```text
V_out_clamp[k] =
    (sum_i g[k,i] * x[i] + (1 / RS_CLAMP) * V_clamp_target[k])
    / (sum_i g[k,i] + 1 / RS_CLAMP)
```

The update keeps the same local contrastive form used by the analog trainer:

```text
param += -gamma * ((dV_clamp^2) - (dV_free^2))
```

with clipping to the same parameter bounds used in the analog code.

Notes:

- the file still uses the saved name `vg_unique` for compatibility with the existing run-file conventions
- in this backend, `vg_unique` is not a transistor gate voltage; it is a clipped linear-resistor parameter mapped to conductance through a fixed scale
- the initial manifest file `netlist_initial.cir` is a descriptive manifest, not an executable SPICE netlist

## Why Vocab32 Was Chosen

The explicit user redirection in this chat was:

- do the 32 vocab script
- use the one-hot CE variant
- do a linear version of that
- run it exactly like it is being run right now but with linear resistors

That is why this work lives around the embed4 one-hot CE trainer rather than around `nonlin_reg` or the digit scripts.

## Output Tree

Linear-resistor vocab32 runs write to:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/`

The first validation run produced a sweep-style example under:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/sweeps/top40_freshproc_p20_20260326-113811/g0.02_d0.05_t0.01/`

That run completed cleanly for 20 epochs. The final epoch summary there reported:

- `val exact acc = 0.3211`
- `val soft ce = 2.5327`
- `val qmass = 0.1263`
- `epoch time ≈ 1.38s`

## Sweep Requested In Chat

The requested 20-epoch sweep was:

- temperatures: `0.001`, `0.01`, `0.1`
- gammas: `0.03`, `0.3`, `3`
- deltas: three reasonable values chosen as `0.05`, `0.1`, `0.2`
- initialization modes:
  - fixed `1.5 V`
  - random uniform between `1 V` and `3 V`

Total combinations:

- `3 * 3 * 3 * 2 = 54`

The estimate given for 5 workers was:

- about `16-18 minutes`

That estimate came from observed run times around `88-90s` per run.

The exact launcher used for that sweep is preserved in:

- `lang_model/vocab32/linear_resistor/run_gdt_init_sweep.py`

The completed sweep root is:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/sweeps/gdt_inits_p5_20260326-232434/`

The launcher log is:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/sweeps/gdt_inits_p5_20260326-232434/launcher.log`

That sweep finished cleanly for all `54/54` runs.

## Best Support Accuracy Mentioned In Chat

At the time the support question was asked, the trainer did not yet log `support_acc`, so support was computed post hoc from saved run artifacts.

The best support accuracy reported so far in that conversation was:

- `0.9407`

for:

- `g=0.3`
- `delta=0.05`
- `temp=0.001`
- init `random 1..3`

The run folder was:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/sweeps/gdt_inits_p5_20260326-232434/g0.3_d0.05_t0.001_init-rand1to3/`

Other top support values mentioned in the same answer were:

- `0.9393` at `g=0.03, d=0.1, T=0.001`, init fixed `1.5`
- `0.9292` at `g=0.03, d=0.05, T=0.001`, init fixed `1.5`

## Logging Changes Added Later

After the sweep and support inspection, the trainer was updated so it now logs support-oriented metrics directly.

The later request was:

- start logging `support_acc`
- log the probability distribution over valid next tokens

That update added:

- empirical `q(.|context)` built from the train split
- `support_acc` for train and validation
- `qmass_mean` measured against the empirical target support
- unseen-context counts for validation
- context-aware preview/sample JSON outputs carrying the valid-next-token distribution

New saved histories now include:

- `0_train_support_acc.npy`
- `0_val_support_acc.npy`
- `0_train_qmass.npy`
- `0_val_qmass.npy`
- `0_val_unseen_contexts.npy`

The controller now saves:

- `train_ctx.npy`

and the worker is backward-compatible with older runs because it can reconstruct train contexts from `train_x.npy` if `train_ctx.npy` is missing.

Important limitation:

- runs that started before this logging patch do not retroactively gain the new support-accuracy histories in their saved artifacts

## Hidden-Path Follow-Up

The next architectural variant requested in chat added another path from every input to every output through a unique hidden node:

- one direct edge `input[i] -> output[k]`
- one extra serial path `input[i] -> hidden[i,k] -> output[k]`

This produced a second trainer:

- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py`

with a wrapper entry point:

- `lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path.py`

and a curriculum-style fine-tune launcher:

- `lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path_hard_examples.py`

Because each hidden node belongs to exactly one input/output pair, it can be eliminated analytically during the output solve. The hidden branch contributes an effective conductance:

```text
g_series[i,k] = (g_in_hidden[i,k] * g_hidden_out[i,k]) / (g_in_hidden[i,k] + g_hidden_out[i,k])
g_total[i,k] = g_direct[i,k] + g_series[i,k]
```

The trainer still applies the same free/clamp local update law, but now separately on:

- the direct edge
- the `input -> hidden` edge
- the `hidden -> output` edge

The hidden-path result tree is:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_hiddenpath_onehotce/`

The hidden-path trainer was later extended with the same fast execution knobs as
the direct trainer:

- `--process-mode in_process`
- `--eval-every`
- `--sample-every`
- `--plot-every`

It also now supports an optional hidden-path edge-remodeling pass directly in
the same trainer:

- `--remodel-every-epochs`
- pruning on low utility / low backbone edges
- rebirth of exactly as many inactive edges as were pruned in that remodeling step

The hard-example fine-tune launcher resumes from a saved hidden-path checkpoint,
mines train rows the checkpoint still fails, and starts a fresh fine-tune run.
Its default behavior is intentionally conservative:

- it picks the best saved hidden-path checkpoint by `val_qmass_mean` if no base run is supplied
- it focuses on `support` misses by default instead of raw `exact` misses, because many contexts have multiple valid next tokens
- it mixes in replay rows from already-correct examples to reduce forgetting
- it writes a new run folder instead of mutating the source run

## Crossbar Remodeling Follow-Up

The direct-crossbar follow-up kept the usual fast local edge update but added a
slow topology pass over the existing `24 -> 32` crossbar.

Because this variant has no hidden-node pool, it only implements edge
birth/death, not node split/merge:

- binary edge activity mask on the direct crossbar
- utility EMA from `|(dV_C^2) - (dV_F^2)|`
- backbone EMA from `g_e * (dV_F^2)`
- pruning when both utility and backbone stay small
- small random disassembly
- rebirth of inactive edges with nonzero initial parameter values when active
  density drops below a target fraction

This produced a third trainer:

- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py`

with wrapper:

- `lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_remodel.py`

and result tree:

- `lang_model/vocab32/results_language_32_embed4_linear_resistor_remodel_onehotce/`

## 50-Epoch Direct vs Hidden-Path Comparison

The best direct hyperparameters chosen from the earlier sweep proxy were:

- `gamma=0.3`
- `delta=0.05`
- `temp=0.001`
- init `random 1..3`

Those settings were rerun for `50` epochs on:

- direct model:
  - `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/comparisons/qmass_best_50ep_direct_g0.3_d0.05_t0.001_init-rand1to3/`
- hidden-path model:
  - `lang_model/vocab32/results_language_32_embed4_linear_resistor_hiddenpath_onehotce/comparisons/qmass_best_50ep_hiddenpath_g0.3_d0.05_t0.001_init-rand1to3/`

Using validation `qmass_mean` as the probability-mass-on-valid-next-token metric:

- direct best within 50 epochs:
  - epoch `14`
  - `val_qmass_mean = 0.9290`
  - `val_support_acc = 0.9555`
  - `val_exact_acc = 0.4139`
- hidden-path best within 50 epochs:
  - epoch `35`
  - `val_qmass_mean = 0.9333`
  - `val_support_acc = 0.9876`
  - `val_exact_acc = 0.4386`

At the final `50`-epoch checkpoint:

- direct final:
  - `val_qmass_mean = 0.7801`
  - `val_support_acc = 0.9383`
  - `val_exact_acc = 0.4077`
- hidden-path final:
  - `val_qmass_mean = 0.9267`
  - `val_support_acc = 0.9600`
  - `val_exact_acc = 0.4198`

Conclusion from this comparison:

- the hidden-path model improved over the direct model on the requested valid-next-token probability metric
- it also improved support accuracy and exact accuracy
- the direct model peaked much earlier and degraded substantially by epoch `50`
- the hidden-path model stayed close to its best value through the end of training

## Sample and Preview Outputs

The updated trainer now writes valid-next-token distribution context to:

- `sample_target_preview.json`
- `samples_epoch*.txt`
- `samples_epoch*.json`

The JSON outputs now include:

- the context
- the observed next token
- `valid_next_token_distribution`
- `valid_next_token_distribution_top`
- generated continuation for the sample files

## Commands

Folder wrapper entry point:

```bash
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce.py 0 --epochs 20
```

Hidden-path wrapper entry point:

```bash
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_hidden_path.py 0 --epochs 20
```

Crossbar remodeling wrapper entry point:

```bash
conda run -n p311env python lang_model/vocab32/linear_resistor/train_embed4_onehot_ce_remodel.py 0 --epochs 20
```

Direct trainer entry point:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py 0 --epochs 20
```

Hidden-path trainer entry point:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py 0 --epochs 20
```

Crossbar remodeling trainer entry point:

```bash
conda run -n p311env python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py 0 --epochs 20
```

Exact 54-run sweep launcher from this chat:

```bash
conda run -n p311env python lang_model/vocab32/linear_resistor/run_gdt_init_sweep.py
```

## Related Files

- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py`
  - analog NMOS embed4 one-hot CE trainer
- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor.py`
  - linear-resistor KCL trainer
- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path.py`
  - linear-resistor KCL trainer with both direct and per-pair hidden paths
- `lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce_linear_resistor_remodel.py`
  - linear-resistor KCL trainer with slow edge remodeling on the direct crossbar
- `lang_model/vocab32/linear_mlp_lang_embed4_onehot_ce.py`
  - linear softmax baseline used as a simple comparison point
- `lang_model/vocab32/results_language_32_embed4_linear_resistor_onehotce/`
  - run artifacts for this backend
- `lang_model/vocab32/results_language_32_embed4_linear_resistor_hiddenpath_onehotce/`
  - run artifacts for the hidden-path backend
- `lang_model/vocab32/results_language_32_embed4_linear_resistor_remodel_onehotce/`
  - run artifacts for the crossbar remodeling backend

## Intent Of This README

This file is not meant to be a polished paper-style summary. It is a compact in-repo handoff of the decisions, equations, run setup, and follow-up logging work that were established in the chat so the next session does not have to reconstruct them from scratch.

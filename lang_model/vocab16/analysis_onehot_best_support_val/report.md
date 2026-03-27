# 16-vocab one-hot CE best-support analysis

## Selected checkpoint
- Run: `g0.05_d0.1_t0.01`
- Epoch: `17`
- Sweep criterion: `support_acc == 1.0`, then highest `val_qmass_mean`
- Selected metrics: `val_qmass_mean = 0.377538`, `val_exact_acc = 0.525362`, `val_soft_ce = 1.403694`

## High-level findings
- The selected checkpoint is the best of `7` support-perfect epochs by `val_qmass_mean`.
- Final validation mean entropy is `2.000` bits, with mean correct-token rank `2.004`.
- Final validation exact accuracy is `0.525`, while support-perfect selection still leaves substantial uncertainty inside the valid set.
- Bigram residual L1 is `5.814` and forbidden bigram probability mass is `2.079`.
- Rollout invalid-transition rate at sampling temperature `0.050` is `1.000`.

## Figure bundle
- `figure1_training_uncertainty.png`: training curves, perplexity, uncertainty, reliability
- `figure2_distributional_competence.png`: empirical/model bigrams, residuals, forbidden-transition leaks
- `figure3_context_distributions.png`: representative contexts plus rank/entropy/top-k summaries
- `figure4_context_use.png`: causal position ablations and token influence matrix
- `figure5_internal_circuits.png`: gate trajectory PCA, gate matrix, probe refinement, mean gate by position
- `figure6_minimal_pairs_calibration_generation.png`: minimal-pair panels, reliability, rollout diagnostics

## Probe contexts
- `<BOS>`: final support mass `0.125`
- `the`: final support mass `1.000`
- `the boy is`: final support mass `0.905`
- `the boy likes the`: final support mass `0.989`

## Strongest final edges
- `<EOS>` <= position `6` bit `1` with `VG = 3.6770`
- `<BOS>` <= position `3` bit `1` with `VG = 3.0869`
- `<EOS>` <= position `4` bit `1` with `VG = 3.0567`
- `<EOS>` <= position `3` bit `2` with `VG = 2.9693`
- `<BOS>` <= position `2` bit `4` with `VG = 2.9062`
- `<BOS>` <= position `1` bit `2` with `VG = 2.7991`
- `ball` <= position `1` bit `2` with `VG = 2.5485`
- `sees` <= position `5` bit `3` with `VG = 2.5431`
- `<BOS>` <= position `6` bit `1` with `VG = 2.4944`
- `ball` <= position `5` bit `1` with `VG = 2.4698`

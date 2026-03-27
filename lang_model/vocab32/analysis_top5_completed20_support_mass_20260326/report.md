# 32-vocab one-hot CE top-5 completed-run analysis

## Selection
- Sweep: `top40_freshproc_p20_20260326-113811`
- Population: runs from March 26, 2026 with completed `epoch 20` and saved `0_vout_val_epoch20.npy`
- Ranking criterion: corrected `support_mass_mean` at epoch 20, then `support_acc`, then `exact_acc`
- Selected runs: `g0.03_d0.05_t0.01, g0.02_d0.05_t0.01, g0.05_d0.05_t0.005, g0.1_d0.05_t0.005, g0.05_d0.05_t0.01`

## Top Run
- Run: `g0.03_d0.05_t0.01`
- Epoch-20 support_mass_mean: `0.733539`
- Epoch-20 support_acc: `0.883717`
- Epoch-20 exact_acc: `0.378342`
- Epoch-20 true_token_prob_mean: `0.249385`
- Epoch-20 soft_ce: `1.858500`

## High-level findings
- The best completed epoch-20 run by corrected support mass is `g0.03_d0.05_t0.01` with `support_mass_mean = 0.734`.
- Across the selected top 5, epoch-20 support mass spans `0.638` to `0.734`.
- Temperature mix in the top 5: `{0.01: 3, 0.005: 2}`.
- Error mass remains concentrated in a few recurring families: subject agreement (`are` for `am`), early question punctuation, bad terminal transitions, and `is` replacing object nouns.
- This report uses corrected support mass over all valid next tokens; it does not use the old single-target `val_qmass_mean` from the original sweep summaries.

## Figure bundle
- `figure1_epoch20_ranking.png`: top-5 epoch-20 ranking by support mass, support accuracy, exact accuracy
- `figure2_metric_space.png`: epoch-20 metric-space scatter for all completed runs, top 5 highlighted
- `figure3_top5_training_curves.png`: support mass, support accuracy, and true-token probability across epochs for the top 5
- `figure4_error_families.png`: epoch-20 invalid-prediction family breakdown for the top 5
- `figure5_support_mass_histograms.png`: validation support-mass distributions for the top 5
- `figure6_low_mass_contexts.png`: lowest-support-mass contexts for each selected run

## Top 5 epoch-20 runs
- `g0.03_d0.05_t0.01`: support_mass_mean `0.733539`, support_acc `0.883717`, exact_acc `0.378342`, true_token_prob_mean `0.249385`, invalid_count `1309`
- `g0.02_d0.05_t0.01`: support_mass_mean `0.729180`, support_acc `0.873679`, exact_acc `0.365017`, true_token_prob_mean `0.254793`, invalid_count `1422`
- `g0.05_d0.05_t0.005`: support_mass_mean `0.680516`, support_acc `0.828995`, exact_acc `0.339877`, true_token_prob_mean `0.233042`, invalid_count `1925`
- `g0.1_d0.05_t0.005`: support_mass_mean `0.662720`, support_acc `0.812828`, exact_acc `0.302834`, true_token_prob_mean `0.220751`, invalid_count `2107`
- `g0.05_d0.05_t0.01`: support_mass_mean `0.637727`, support_acc `0.850937`, exact_acc `0.357911`, true_token_prob_mean `0.216787`, invalid_count `1678`

## Dominant error families by run
- `g0.03_d0.05_t0.01`: other=623, early_question_mark=300, are_for_am=271, bad_terminal_transition=75
- `g0.02_d0.05_t0.01`: other=640, early_question_mark=442, are_for_am=271, bad_terminal_transition=65
- `g0.05_d0.05_t0.005`: other=968, early_question_mark=413, are_for_am=271, repeat_or_stall=152
- `g0.1_d0.05_t0.005`: other=1394, early_question_mark=671, bad_terminal_transition=42
- `g0.05_d0.05_t0.01`: other=768, early_question_mark=303, are_for_am=271, is_for_object_noun=116

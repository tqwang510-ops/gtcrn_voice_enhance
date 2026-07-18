# Runs artifact index

`runs/` is ignored by Git because it contains checkpoints, generated audio, plots, and
large evaluation outputs. This index records the small result artifacts that are useful
for reproducing the conclusions in `TRAINING_PROTOCOL.md`.

## v5 evaluation

- Evaluation outputs: `runs/v5_eval/`
- Listening matrix: `runs/v5_eval/listening/`
- Listening manifest: `runs/v5_eval/listening_manifest.json`
- Command logs: `runs/v5_eval/logs/`
- Validation-domain baselines: `runs/v5_eval/validation_baselines/`

## Validation baselines

The files whose names contain `old_snr18_32` were produced before the v5 mixtures were
regenerated with the final 8-22 dB background-SNR distribution. They are retained for
history and must not be compared directly with the final v5 model.

- `v3_old_snr18_32.json`: v3 on the original quieter v5 validation set.
- `v4_old_snr18_32.json`: v4 on the original quieter v5 validation set.
- `repair_old_snr18_32.json`: identity-repair model on the original quieter set.
- `v3_current_snr8_22.json`: v3 on the final v5 validation set, using the original
  first-N validation selection.

New baseline evaluations made after the validation-accounting repair should include
`sampled` in their file name. Those runs use deterministic random sampling configured by
`sample_seed` in `validation_domains_v5.json`.

- `v3_current_sampled.json`: v3, final SNR data, repaired validation sampling.
- `v4_current_sampled.json`: v4, final SNR data, repaired validation sampling.
- `v5_current_sampled.json`: v5 epoch 3 (`best.tar`) on the same files.
- `v5_epoch5_current_sampled.json`: v5 epoch 5 candidate on the same files.

## Epoch 5 candidate audit

- Full v5 Chinese test: `runs/v5_epoch5_eval/v5_test/`
- Classroom listening matrix: `runs/v5_epoch5_eval/listening/`
- AISHELL normalized clean comparison: `runs/v5_epoch5_eval/aishell_clean_norm_compare/`
- Epoch 3 clean listening: `runs/v5_epoch5_eval/aishell_clean_norm_compare/listening_epoch3/`
- Epoch 5 clean listening: `runs/v5_epoch5_eval/aishell_clean_norm_compare/listening_epoch5/`

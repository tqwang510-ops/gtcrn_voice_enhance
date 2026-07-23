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
- `v5_on_v6_baseline.json`: v5 epoch 3 baseline on the final v6 eight-domain
  validation protocol.

## Epoch 5 candidate audit

- Full v5 Chinese test: `runs/v5_epoch5_eval/v5_test/`
- Classroom listening matrix: `runs/v5_epoch5_eval/listening/`
- AISHELL normalized clean comparison: `runs/v5_epoch5_eval/aishell_clean_norm_compare/`
- Epoch 3 clean listening: `runs/v5_epoch5_eval/aishell_clean_norm_compare/listening_epoch3/`
- Epoch 5 clean listening: `runs/v5_epoch5_eval/aishell_clean_norm_compare/listening_epoch5/`

## v6 denoise evaluation

- Training run: `runs/classroom_v6_denoise/`
- Selected candidate: `runs/classroom_v6_denoise/checkpoints/candidate_epoch_009.tar`
- Full evaluation: `runs/v6_eval/`
- v5 comparison listening: `runs/v6_eval/listening_v5_baseline/`
- v6 epoch 9 listening: `runs/v6_eval/listening_epoch9/`

## v7 continuous and student-murmur evaluation

- Training run: `runs/classroom_v7/`
- Selected checkpoint: `runs/classroom_v7/checkpoints/best.tar` (epoch 15)
- v7 test evaluation: `runs/v7_eval/best_test/`
- v5 baseline on the same test: `runs/v7_eval/v5_test/`
- Listening manifest: `runs/v7_eval/listening_manifest.json`
- v5 listening output: `runs/v7_eval/listening/01_v5/`
- v7 listening output: `runs/v7_eval/listening/02_v7_best/`

## v7.1 stronger-noise evaluation

- Training run: `runs/classroom_v7_1/`
- Selected checkpoint: `runs/classroom_v7_1/checkpoints/best.tar` (epoch 12)
- v7.1 test evaluation: `runs/v7_1_eval/v7_1_test/`
- v5 baseline on the same test: `runs/v7_1_eval/v5_on_v7_1_test/`
- v4 regression: `runs/v7_1_eval/v4_test/` (pending at documentation time)
- v7.1 command/generation/evaluation logs: `runs/v7_1_eval/logs/`
- v7.1 audit/provenance JSON: `runs/v7_1_eval/provenance/`
- v7.1 listening manifest: `runs/v7_1_eval/listening_manifest.json`
- v5 listening output: `runs/v7_1_eval/listening/01_v5/`
- v7.1 listening output: `runs/v7_1_eval/listening/02_v7_1_best/`

## v7.2 low-SNR speech-preservation smoke

- Smoke dataset: `../dataset_classroom_v7_2_smoke/generated/`
- Validation config: `validation_domains_v7_2_smoke.json`
- v7.1 initialization baseline: `runs/v7_2_eval/provenance/v7_2_smoke_baseline.json`
- Reproduced training output: `runs/classroom_v7_2_smoke/` (best = epoch 1)
- Same-file v7.1 comparison: `runs/v7_2_eval/listening/01_v7_1/`
- Same-file v7.2 comparison: `runs/v7_2_eval/listening/02_v7_2_epoch1/`
- Thirteen-file metrics: `runs/v7_2_eval/listening13_metrics/`
- Causal gain-smoothing audition: `runs/v7_2_eval/listening_gain_smoothing/`

## Completed v7.1 regressions

- v4 test: `runs/v7_1_eval/v4_test/summary.json`
- v2 test: `runs/v7_1_eval/v2_test/summary.json`
- VoiceBank test: `runs/v7_1_eval/voicebank_test/summary.json`

# 冻结发布：GTCRN classroom v7.2 epoch 1

**冻结日期**：2026-07-23
**状态**：主观最佳候选，已完成完整离线回归与扩大试听验收；待流式一致性验证和香橙派实测后才算部署验收。

## 发布标识

```text
文件：   release/gtcrn_classroom_v7_2_epoch1.tar
SHA256： 6f38816cf6d31a3578c699986d89b15101efc70ca1e4fdb6025a66dce24b472a
来源：   runs/classroom_v7_2_smoke/checkpoints/best.tar（epoch 1，已校验一致）
Git：    2f91172ad14ec6f7886871e34664832a54ed7e28（冻结时协议文档有未提交修改）
```

不要使用 `runs/**/best.tar` 作为发布标识，该路径可能被后续训练覆盖。

## 模型配置

```text
结构：GTCRN（n_fft=256，31,861 参数）
fs=16000, win_length=160 (10 ms), hop_length=80 (5 ms), n_fft=256, center=true
输入：带噪复数 STFT；输出：复数掩码增强 STFT
```

## 训练血缘与本轮训练

```text
voicebank_serious_v1 -> classroom_v2 -> classroom_replay_v3 -> classroom_v5_chinese
-> classroom_v7_1 (epoch 12) -> classroom_v7_2_smoke (epoch 1，本发布)
```

v7.2 为 3 epoch repair 微调：从 v7.1 best 初始化，lr 5e-6，seed 20260730，
`--far-preserve-rir-max-snr 6`，`--speech-underestimate-weight 1.0`，
25% VoiceBank replay，frozen BN。

## 验收记录

- 完整离线回归（17.32，与 v7.1 epoch 12 同文件对比）：六个评估全部持平或更好；
  v7.1 test SI-SNR +3.02 dB / PESQ +0.365 / improved 95.6%；clean 透传全面改善
  （AISHELL raw 112.3 dB / norm 90.3 dB）。输出 `runs/v7_2_eval/full_regression/`。
- 扩大试听矩阵（31 组，`runs/v7_2_eval/listening_expanded/`）：用户确认低 SNR
  持续噪声降噪好、人声还原度高；无 v7.2 新增人声损伤，残留损伤两模型共有且
  听不出明显差别。第 2 步通过。

## 已知限制（接受并记录，不在本轮解决）

- 低 SNR 持续机器噪声（如 04_machine_low）有残余噪声和轻微时变调制（音乐噪声
  质感），v7.1/v7.2/官方 DNS3 均未能完全消除，判定为 GTCRN 复数掩码路线上限。
- 非持续声（学生讨论、瞬态）降噪不干净；test_000504 / test_000027 两个高 SNR
  left-tail 样本两模型 SI-SNR 均约 -7 dB。
- 整机联调兜底手段：输出混合比例、干净语音旁路、增益限制（见
  `gtcrn_project_timeline.md` 节点 7）。

## 回退版本

```text
runs/classroom_v7_1/checkpoints/best.tar（epoch 12，正式父模型）
香橙派链路原版 GTCRN（官方权重，设备端现有版本）
```

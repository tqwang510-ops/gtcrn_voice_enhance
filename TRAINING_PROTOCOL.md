# GTCRN Serious Training Protocol

## 1. Purpose

This protocol replaces the earlier minimal learning run. The serious run must
be reproducible, must not use the official VoiceBank test set to choose a
checkpoint, and must save enough information for later analysis.

The GTCRN network features and HybridLoss remain based on the original
repository. The engineering changes cover data sampling, split control,
learning-rate scheduling, experiment logging, checkpointing, and evaluation.

重要的代码来源边界：

```text
GTCRN 官方仓库提供：
  模型结构（gtcrn.py）
  HybridLoss 损失函数（loss.py）
  DNS3 和 VCTK-DEMAND 预训练权重
  离线推理示例（infer.py）
  流式模型转换示例（stream/）

本项目后来增加：
  成对 noisy/clean wav 的读取和裁剪
  train/validation/test 划分
  优化器和学习率计划
  epoch 循环、日志、最优模型选择和断点续训
  自定义评估和教室数据生成
```

官方仓库没有提供可直接执行的完整训练入口。官方 README 只是链接到另一个
`SEtrain` 仓库，把它作为通用语音增强训练模板。因此，本文档中的优化器、
warmup、50 个 epoch、数据划分和片段选择方法，都是本项目为了可复现而制定的
工程方案，不能称为 GTCRN 作者公布的原始完整训练配置。

## 2. Signal configuration

```text
sample rate: 16000 Hz
window length: 160 samples (10 ms)
hop length: 80 samples (5 ms)
FFT size: 256
frequency bins: 129
STFT center: true
segment length: 2 seconds
```

当前离线模型质量实验继续使用 `center=true`。严格因果的流式音频前端属于另一
个部署实验。不能只改这个开关，还必须一起检查分析窗、合成窗、重叠相加、帧
对齐和输出长度。

### 2.1 模型训练到底在做什么（初学者说明）

一个训练样本由时间严格对应的两个文件组成：

```text
noisy wav：麦克风收到的语音，包含噪声和混响
clean wav：希望模型输出的目标，必须是同一句话和同一时间区间
```

训练程序会不断重复下面的过程：

```text
1. 读取文件名对应的 noisy/clean 音频对。
2. 从两份音频中选择完全相同的 2 秒区间。
3. 把两份波形都转换成复数 STFT 时频帧。
4. 把 noisy STFT 输入 GTCRN。
5. GTCRN 预测复数掩码，并得到增强后的 STFT。
6. HybridLoss 同时在频域和波形域比较增强结果与 clean 目标。
7. PyTorch 反向传播计算每个模型参数对误差的影响，Adam 再更新参数，使以后
   的误差尽量变小。
8. 完整遍历一次训练集称为一个 epoch。每个 epoch 结束后只做验证、不更新
   参数；验证损失更低时保存为 `best.tar`。
```

所以，训练并不是“把音频扔进去让模型自己随便学”。成对数据中的 clean 目标
明确规定了模型应该保留什么、去掉什么。对于去混响，使用完全干声作为标签，
还是使用“直达声加早期反射”作为标签，本质上是两个不同任务。

### 2.2 `center` 到底是什么意思

`center` 是 PyTorch 在做 STFT 分帧时的参数，不是 GTCRN 神经网络内部的开关。

`center=true` 时，PyTorch 会在波形边界补数据，并把第 `t` 帧定义为大致以
`t * hop_length` 样本为中心。这适合整段文件的离线处理，也更容易重建开头和
结尾。相对于这一帧标注的中心时刻，分析窗会使用中心两侧的样本，因此实时系统
必须等所需样本到齐。这属于有限的缓冲延迟，并不会把时间方向的 GRU 变成双向。

`center=false` 时，PyTorch 不再进行这种居中补齐。帧的位置和输出长度会变化，
合成阶段还必须保证信号开头、结尾满足重叠相加条件。它适合用来设计因果流式
前端，但不是一个可以直接打开的“低延迟模式”。

当前项目使用的参数是：

```text
fs = 16000
n_fft = 256
win_length = 160
hop_length = 80
window = square-root Hann
```

2026-07-15 已经用上述参数做过直接往返测试。设置 `center=false` 后，
`torch.istft` 会报告 window overlap-add 错误。当前 `HybridLoss` 也会调用这
个 ISTFT，所以现在仅仅给训练命令增加 `--no-center`，结果不是得到流式模型，
而是训练直接报错。

### 2.3 现在需要使用 `center=false` 重新训练吗？

当前结论：**不需要。下一轮模型质量实验继续使用 `center=true`。**

原因如下：

```text
1. 官方发布的 infer.py 和原始 HybridLoss 都没有传 center 参数，因此使用的
   是 PyTorch 默认值 center=true。
2. 已经完成的 VoiceBank 基线及其 checkpoint 使用 center=true。
3. 当前 center=false 的波形重建并不能正常工作。
4. 目前更重要的是解决房间/噪声跨集合泄漏和 clean 目标定义问题，而不是先改
   STFT 帧对齐方式。
```

部署时应另外建立一个流式实验：

```text
1. 先保持相同的 fs/n_fft/win_length/hop_length 和 square-root Hann 窗。
2. 明确实现输入缓冲、STFT 取帧、循环网络/卷积缓存、重叠相加、启动和结束
   flush 行为。
3. 用同一个 wav 比较流式输出和离线输出。
4. 在目标设备上测量端到端延迟和实时率。
5. 只有当流式前端明显改变输入特征分布或音质时，才使用完全相同的流式前端
   进行微调或重新训练。
```

不要把 `center=false` 直接等同于“因果模型”。GTCRN 的时间方向卷积和帧间 GRU
按因果方式设计，`stream` 目录又为卷积和循环层增加了显式缓存。STFT 缓冲、
神经网络因果性和整个系统延迟互相关联，但不是同一件事。

### 2.4 当前能够确认的 GTCRN 原版训练信息

从本地官方仓库能够确认：

```text
数据集：VCTK-DEMAND 和 DNS3
infer.py 采样率：16 kHz
原版 STFT：n_fft=512、win_length=512、hop_length=256
原版 center：true（PyTorch 默认值）
输入：带噪复数 STFT
输出：通过复数比例掩码得到的增强复数 STFT
损失：压缩后的实部/虚部频谱 MSE + 压缩幅度 MSE + 波形 SI-SNR 项
权重：model_trained_on_vctk.tar 和 model_trained_on_dns3.tar
```

检查两个官方 checkpoint 后还可以确认：它们都保存了 Adam 优化器状态，参数为
`betas=(0.9, 0.999)`、`eps=1e-8`、weight decay 为 0。VCTK-DEMAND 权重
保存在 epoch 92，DNS3 权重保存在 epoch 87；其中保存的学习率是 `3.125e-5`。
但是只看最终 checkpoint，不能可靠推出初始学习率和完整的学习率变化过程。

本地官方仓库**不能确认**作者训练时准确使用的 batch size、片段长度、完整混音
方法、初始学习率、scheduler、验证集划分和 checkpoint 选择规则。这些细节必须
继续以论文或官方链接的 `SEtrain` 配置为依据，才能称为官方方案。当前
`train_custom.py` 是与 GTCRN 结构兼容的自定义训练实现，不是从作者原始训练
代码中恢复出来的副本。

## 3. Fixed dataset split

Run:

```powershell
python make_voicebank_splits.py --dataset-root ..\dataset --output-dir ..\dataset\splits\voicebank_serious --seed 42 --valid-speaker-fraction 0.1
```

The generated speaker-disjoint split is:

```text
train: 25 speakers, 10,235 paired files
valid:  3 speakers,  1,337 paired files
test:   VoiceBank official test set, 824 paired files
```

Validation speakers are fixed to:

```text
p250, p268, p270
```

The official test set is not used during training or checkpoint selection.
It is evaluated only after training is complete.

Manifests:

```text
..\dataset\splits\voicebank_serious\train.json
..\dataset\splits\voicebank_serious\valid.json
..\dataset\splits\voicebank_serious\test.json
```

## 4. Segment and silence policy

For each training file, the clean and noisy wav files use exactly the same
start and end sample. Up to 10 random candidate segments are tried. A segment
with clean RMS at or above `-40 dBFS` is accepted; if all candidates are below
the threshold, the highest-energy candidate is used.

Validation is deterministic. It examines 16 fixed candidate positions and
always selects the highest-energy one. Therefore the validation segment for a
file does not change between epochs.

The final test evaluation uses complete wav files without random cropping.

This gate prioritizes speech-bearing segments. It does not replace a real VAD,
and a future deployment dataset should deliberately include a controlled
amount of clean-silence/noisy-background examples with an appropriate loss.

## 5. Optimization

```text
optimizer: Adam
maximum learning rate: 1e-3
weight decay: 0
gradient clipping: 5.0
warmup: 3 epochs, from 1e-6 to 1e-3
decay: cosine annealing to 1e-5
seed: 42
```

Warmup limits large updates while the randomly initialized model is unstable.
Cosine decay keeps larger steps early and progressively reduces the update
size for later refinement. It is an optimization strategy, not a substitute
for correct data or evaluation.

## 6. Training command

Start this experiment from random initialization. Do not resume the earlier
minimal-run checkpoint because the split, validation policy, scheduler, and
experiment records have changed.

```powershell
cd D:\modeltraining\gtcrn
python train_custom.py --train-noisy ..\dataset\train\noisy --train-clean ..\dataset\train\clean --valid-noisy ..\dataset\train\noisy --valid-clean ..\dataset\train\clean --train-manifest ..\dataset\splits\voicebank_serious\train.json --valid-manifest ..\dataset\splits\voicebank_serious\valid.json --out-dir runs\voicebank_serious_v1 --epochs 50 --batch-size 4 --lr 1e-3 --scheduler warmup_cosine --warmup-epochs 3 --warmup-start-lr 1e-6 --min-lr 1e-5 --min-clean-rms-db -40 --seed 42
```

Resume after an interruption:

```powershell
python train_custom.py --train-noisy ..\dataset\train\noisy --train-clean ..\dataset\train\clean --valid-noisy ..\dataset\train\noisy --valid-clean ..\dataset\train\clean --train-manifest ..\dataset\splits\voicebank_serious\train.json --valid-manifest ..\dataset\splits\voicebank_serious\valid.json --out-dir runs\voicebank_serious_v1 --epochs 50 --batch-size 4 --lr 1e-3 --scheduler warmup_cosine --warmup-epochs 3 --warmup-start-lr 1e-6 --min-lr 1e-5 --min-clean-rms-db -40 --seed 42 --resume runs\voicebank_serious_v1\checkpoints\last.tar
```

## 7. Saved experiment artifacts

```text
runs\voicebank_serious_v1\config.json
runs\voicebank_serious_v1\metrics.csv
runs\voicebank_serious_v1\training_curve.png
runs\voicebank_serious_v1\checkpoints\best.tar
runs\voicebank_serious_v1\checkpoints\last.tar
```

`metrics.csv` is the source record. The curve image is derived from the CSV
and is regenerated after every epoch.

## 8. Final test evaluation

After training, evaluate only `best.tar` on the untouched official test set:

```powershell
python evaluate_custom.py --checkpoint runs\voicebank_serious_v1\checkpoints\best.tar --noisy-dir ..\dataset\valid\noisy --clean-dir ..\dataset\valid\clean --manifest ..\dataset\splits\voicebank_serious\test.json --out-dir runs\voicebank_serious_v1\evaluation --save-hardest 10
```

Outputs:

```text
evaluation\metrics.csv
evaluation\summary.json
evaluation\hardest_samples\*_noisy.wav
evaluation\hardest_samples\*_enhanced.wav
evaluation\hardest_samples\*_clean.wav
```

The report includes input, enhanced, and improvement values for:

```text
SI-SNR
wideband PESQ
STOI
```

Objective scores must be considered together with listening tests. For the
ceiling-microphone product, final acceptance still requires real room,
reverberation, loudspeaker leakage, and feedback-path recordings.

## 9. Implementation verification

The serious pipeline was smoke-tested on 2026-07-14 with 8 training files and
4 validation files. This test validates code paths only; its model scores are
not meaningful quality results.

Verified behavior:

```text
new complex STFT equals the previous real/imag STFT representation exactly
validation segment selection is deterministic for the same file
training segment selection changes across epochs
GPU training, warmup, CSV logging, curve output, and checkpoint output work
checkpoint resume continues from the next epoch and retains optimizer state
SI-SNR, PESQ-WB, STOI, CSV, JSON, and audio-sample evaluation outputs work
```

The earlier minimal-run checkpoint remains separate in `checkpoints_custom/`.
It is not the starting point for the serious experiment.

## 10. VoiceBank baseline result on 2026-07-14

This run is a baseline noise-suppression experiment, not the final ceiling
microphone / local PA model.

Run directory:

```text
runs\voicebank_serious_v1
```

Best checkpoint:

```text
runs\voicebank_serious_v1\checkpoints\best.tar
best epoch: 47
```

Full test evaluation:

```text
runs\voicebank_serious_v1\eval_test_full_combined\summary.json
runs\voicebank_serious_v1\eval_test_full_combined\metrics.csv
```

Summary on the 824-file official VoiceBank test set:

```text
mean input SI-SNR:       8.45 dB
mean enhanced SI-SNR:   17.78 dB
mean SI-SNR improvement: +9.33 dB
improved file fraction:  99.76%

mean input PESQ-WB:      1.97
mean enhanced PESQ-WB:   2.63
mean PESQ improvement:   +0.67

mean input STOI:         0.921
mean enhanced STOI:      0.936
mean STOI improvement:   +0.015
```

Listening samples:

```text
runs\voicebank_serious_v1\eval_test_quick100\hardest_samples
```

Each sample triplet contains:

```text
*_noisy.wav
*_enhanced.wav
*_clean.wav
```

The evaluation script was updated so that 48 kHz test wav files can be
resampled to the 16 kHz model sample rate during evaluation. It also writes
metrics incrementally and supports `--start-index`, which makes long
evaluations more robust when the PESQ extension exits unexpectedly.

## 11. Next data plan for ceiling microphone local PA

The product target is not only generic denoising. The target signal chain is
closer to:

```text
talker speech
-> room reverberation
-> ceiling microphone pickup
-> local amplification / loudspeaker playback leakage
-> possible feedback build-up near howling
-> background noise
```

Therefore the next training data should not be only VoiceBank-DEMAND. Use
VoiceBank as a learning baseline, then build a scenario dataset with the
following components.

### 11.1 Clean speech

Purpose: the clean speech is the supervised training target.

Useful public sources:

```text
AISHELL-1: Chinese Mandarin clean speech, 16 kHz, Apache 2.0.
LibriSpeech: English read speech, about 1000 hours at 16 kHz, CC BY 4.0.
DNS Challenge clean speech: large multi-language clean speech material.
```

Recommendation:

```text
Chinese product first: use AISHELL-1 plus your own Chinese speech recordings.
English data can still help robustness, but it should not replace Chinese data.
```

### 11.2 Room impulse responses and room noise

Purpose: simulate the ceiling microphone receiving reverberant speech.

Useful public sources:

```text
OpenSLR SLR28 RIRs and noises:
  real and simulated room impulse responses
  isotropic and point-source noises
  16 kHz audio

DNS Challenge impulse responses and noise:
  includes clean speech, noise, and room impulse responses
  includes speakerphone / non-headset scenarios
```

Use RIR convolution to generate:

```text
clean_target = dry speech or lightly reverberant direct speech
noisy_input  = speech convolved with room RIR + noise
```

For the ceiling microphone case, the RIR distribution should include:

```text
short RT60 rooms: 0.2-0.4 s
medium RT60 rooms: 0.4-0.8 s
longer rooms: 0.8-1.2 s
talker distances: 0.5-5 m
ceiling-mic-like pickup, if measured data is available
```

### 11.3 Background noise

Purpose: cover HVAC, projector, fan, computer, classroom, conference room,
street leakage, handling noise, and electrical noise.

Useful public sources:

```text
MUSAN: music, speech, and noise recordings.
DNS Challenge noise_fullband: large noise collection.
VoiceBank-DEMAND: still useful as a small benchmark, but not enough alone.
```

Recommendation:

```text
Use public noise for diversity.
Record your own ceiling-mic room noise for realism.
```

### 11.4 Loudspeaker leakage / local PA playback noise

Purpose: model sound from the local loudspeaker returning to the ceiling mic.
This is closer to acoustic echo / leakage than ordinary background noise.

Public datasets can help with general echo behavior, but the real product
needs your own measurements because the loudspeaker, ceiling microphone,
room, mounting position, and gain are product-specific.

Data to record or generate:

```text
play speech/music/noise from the product loudspeaker
record it at the ceiling microphone
use several gains below feedback
move the talker and loudspeaker positions
record with and without near-end speech
record different rooms
```

If the model has only one microphone signal and no loudspeaker reference
signal, this should be treated as leakage/noise suppression. If the runtime
can also provide the loudspeaker playback signal as a reference, the problem
becomes closer to AEC and should use a different model interface.

### 11.5 Feedback / howling edge data

Purpose: expose the model to tonal build-up before or during feedback.

This part is the least likely to be solved by a public dataset. It should be
generated and measured by the product team.

Record safely:

```text
start from low gain
increase gain slowly until just before howling
record short clips near the edge
record several rooms and mic/speaker positions
avoid long high-level howling to protect speakers and ears
label the gain, room, mic position, speaker position, and whether howling starts
```

Synthetic generation can supplement measurements:

```text
choose a feedback-path impulse response h_feedback
feed enhanced/playback signal through h_feedback
apply loop gain below or near 1.0
add narrowband resonant peaks at likely feedback frequencies
mix with reverberant speech and room noise
```

Important: feedback cancellation is not exactly the same as ordinary denoising.
If the system is already unstable, a post-filter speech enhancement model may
not be enough. The deployment system may still need gain control, notch
filtering, adaptive feedback cancellation, or a loudspeaker-reference path.

### 11.6 Recommended next dataset layout

Create a second dataset instead of overwriting VoiceBank:

```text
..\dataset_ceiling_pa\
  clean_speech\
  rir\
  noise\
  loudspeaker_leakage\
  feedback_edge\
  generated\
    train\
      clean\
      noisy\
    valid\
      clean\
      noisy\
    test\
      clean\
      noisy\
  metadata\
```

For each generated pair, save metadata:

```text
clean_file
rir_file
noise_file
snr_db
rt60_estimate
speech_level_dbfs
leakage_gain_db
feedback_gain
room_id
split
random_seed
```

### 11.7 Practical priority

Do this in phases:

```text
Phase A: public clean speech + public RIR + public noise.
Phase B: add your own room noise and ceiling microphone recordings.
Phase C: add loudspeaker leakage measurements.
Phase D: add feedback-edge recordings and synthetic feedback simulation.
Phase E: train and evaluate against a held-out real room test set.
```

Do not train the final product model only on synthetic data. Synthetic data is
useful for scale, but the acceptance test must include real ceiling microphone
recordings from rooms that were not used for training.

### 11.8 Synthetic AISHELL + MUSAN + RIR generator

Script:

```text
make_ceiling_pa_dataset.py
```

Inputs currently available under `D:\modeltraining\dataset`:

```text
data_aishell\wav
musan
RIRS_NOISES
```

Smoke test command:

```powershell
python make_ceiling_pa_dataset.py --out-root ..\dataset_ceiling_pa_smoke\generated --num-train 8 --num-valid 2 --num-test 2 --log-interval 1 --overwrite
```

Recommended first real generated dataset:

```powershell
python make_ceiling_pa_dataset.py --out-root ..\dataset_ceiling_pa\generated --num-train 10000 --num-valid 1000 --num-test 1000 --seed 20260714
```

The generator creates paired files:

```text
generated\train\clean
generated\train\noisy
generated\valid\clean
generated\valid\noisy
generated\test\clean
generated\test\noisy
generated\metadata
```

Original synthetic recipe:

```text
clean target: AISHELL dry speech segment
input speech: clean speech convolved with a simulated RIR
noise: MUSAN noise/music/speech or RIRS pointsource noise
SNR range: 0-20 dB
segment length: 2 seconds
sample rate: 16 kHz
```

This is still a first-stage synthetic dataset. It adds Chinese speech,
reverberation, and background noise. It does not yet contain measured
loudspeaker leakage or feedback-edge recordings.

Important correction on 2026-07-15:

```text
The original broad noise recipe is too wide for an indoor ceiling-microphone
local-PA product. It can include music, external speech, and point-source
noises that may not occur in the target room. Do not use the broad recipe as
the main product-training dataset.
```

The generator now supports `--noise-profile`:

```text
indoor:
  MUSAN/noise
  RIRS_NOISES/real_rirs_isotropic_noises files with "noise" in the name

broad:
  MUSAN/noise + MUSAN/music + MUSAN/speech
  RIRS_NOISES/pointsource_noises

rir_noise_only:
  only RIRS_NOISES/real_rirs_isotropic_noises files with "noise" in the name
```

Recommended indoor regeneration command:

```powershell
python make_ceiling_pa_dataset.py --out-root ..\dataset_ceiling_pa_indoor_v1\generated --num-train 10000 --num-valid 1000 --num-test 1000 --seed 20260715 --noise-profile indoor
```

Use this indoor dataset for the next product-oriented training run instead of
the earlier broad `dataset_ceiling_pa` run.

### 11.9 Train on the first synthetic ceiling-PA dataset

After generating:

```text
..\dataset_ceiling_pa\generated
```

start a new run. Do not use `--resume` from the VoiceBank run, because resume
also restores the old optimizer, epoch number, and run history. Use
`--init-checkpoint` to load only the VoiceBank model weights and start a new
optimizer from epoch 1:

```powershell
python train_custom.py --train-noisy ..\dataset_ceiling_pa\generated\train\noisy --train-clean ..\dataset_ceiling_pa\generated\train\clean --valid-noisy ..\dataset_ceiling_pa\generated\valid\noisy --valid-clean ..\dataset_ceiling_pa\generated\valid\clean --out-dir runs\ceiling_pa_synth_v1 --epochs 50 --batch-size 4 --lr 3e-4 --scheduler warmup_cosine --warmup-epochs 3 --warmup-start-lr 1e-6 --min-lr 1e-5 --min-clean-rms-db -45 --seed 20260714 --init-checkpoint runs\voicebank_serious_v1\checkpoints\best.tar
```

The lower `3e-4` learning rate is intentional for fine-tuning. If training
from random initialization instead, use `--lr 1e-3` and omit
`--init-checkpoint`.

Result on 2026-07-15:

```text
run directory: runs\ceiling_pa_synth_v1
best checkpoint: runs\ceiling_pa_synth_v1\checkpoints\best.tar
best epoch: 49
training curve: runs\ceiling_pa_synth_v1\training_curve.png
test summary: runs\ceiling_pa_synth_v1\eval_test\summary.json
test metrics: runs\ceiling_pa_synth_v1\eval_test\metrics.csv
listening samples: runs\ceiling_pa_synth_v1\eval_test\hardest_samples
```

Synthetic test-set summary:

```text
test files: 1000
files used for summary: 999

mean input SI-SNR:        -1.93 dB
mean enhanced SI-SNR:     -0.21 dB
mean SI-SNR improvement:  +1.72 dB
median SI-SNR improvement: +1.22 dB
improved file fraction:   86.29%

mean input PESQ-WB:       1.39
mean enhanced PESQ-WB:    1.68
mean PESQ improvement:    +0.30

mean input STOI:          0.727
mean enhanced STOI:       0.777
mean STOI improvement:    +0.050
```

Interpretation:

```text
The synthetic ceiling-PA task is harder than the VoiceBank baseline because the
input contains reverberation plus noise while the target is dry AISHELL speech.
The first synthetic run improves most files, especially intelligibility, but it
is not yet a final product model. The next decision should be based on listening
to enhanced/noisy/clean triplets and then adding measured room, loudspeaker
leakage, and feedback-edge data.
```

References checked on 2026-07-14:

```text
DNS Challenge: https://github.com/microsoft/DNS-Challenge
OpenSLR SLR28 RIR/noise: https://www.openslr.org/28/
MUSAN: https://www.openslr.org/17/
LibriSpeech: https://www.openslr.org/12/
AISHELL-1: https://www.openslr.org/33/
```

## 12. 当前教室数据方案：classroom_v2（2026-07-15）

本节是当前应执行的方案，取代 11.8 和 11.9 中基于旧 RIRS_NOISES、宽噪声池
和 `dataset_ceiling_pa` 路径的命令。旧内容仅保留为实验历史。

### 12.1 数据源

```text
clean speech:
  VoiceBank 训练说话人用于 train/valid
  VoiceBank 官方测试说话人用于 test

RIR:
  dataset\BUT_ReverbDB_rel_19_06_RIR-Only
  只读取 RIR 目录内的 IR_sweep*.wav
  共识别 9 个真实房间、2325 个 RIR

noise:
  dataset\PRESTO\ch01.wav ... ch16.wav
  dataset\PCAFETER\ch01.wav ... ch16.wav
  不使用 OOFFICE 和 OMEETING
```

BUT ReverbDB 中还包含 `silence_16kHz_60sec*.wav` 等录音。它们不是脉冲响应，
不能用于 RIR 卷积。生成器已经通过目录名和文件名过滤，只接受真正的
`RIR\IR_sweep*.wav`。

### 12.2 train/valid/test 隔离

生成器先划分数据源，然后生成混音，不再从一个公共池中随机抽样：

```text
speaker: 不同 split 使用不同说话人
RIR:     以 BUT 顶层 room_id 为单位划分，房间之间不重叠
noise:   PRESTO/PCAFETER 的通道 wav 文件之间不重叠
time:    train 只取噪声录音 0%-70%
         valid 只取噪声录音 70%-85%
         test 只取噪声录音 85%-100%
```

PRESTO/PCAFETER 的 16 个通道来自同一段同步录音，因此“通道文件不重叠”仍不
等于“真实声景完全独立”。增加时间区间隔离可以减少内容泄漏，但最终可信测试
仍必须使用未参与训练的真实教室录音。

使用 `split_seed=20260715` 时，BUT 房间固定划分为：

```text
train:
  VUT_FIT_C236
  VUT_FIT_L227
  VUT_FIT_L212
  VUT_FIT_L207
  VUT_FIT_Q301

valid:
  VUT_FIT_E112
  Hotel_SkalskyDvur_ConferenceRoom2

test:
  VUT_FIT_D105
  Hotel_SkalskyDvur_Room112
```

### 12.3 场景比例和标签

`classroom_v2` 按以下概率生成：

```text
10% clean:             输入和标签都是干净语音
15% reverb_only:       输入为完整混响，标签为早期反射目标
60% reverb_noise:      输入为完整混响加小声噪声，标签为早期反射目标
10% noise_no_reverb:   输入为干净语音加小声噪声，标签为干净语音
 5% noise_only:        输入只有背景噪声，标签为全零
```

当前去混响目标不是完全干声，而是保留直达声之后前 50 ms 的早期反射，主要去除
晚期混响。这个目标更符合“去除一些混响”，也通常比强制输出完全干声更自然。

噪声采用 `quiet_classroom` SNR 分布，使 PRESTO/PCAFETER 作为较小背景声：

```text
75%: 20-30 dB SNR
20%: 15-20 dB SNR
 5%: 10-15 dB SNR
```

metadata 记录 `scene_type、target_mode、room_id、noise_class、SNR、估计 RT60、
估计 DRR、speech_activity、seed` 等字段。RT60/DRR 是由 RIR 自动估计的分析值，
不是 BUT ReverbDB 官方标注。

### 12.4 生成正式数据

从 `D:\modeltraining\gtcrn` 执行：

```powershell
D:\Anaconda\Scripts\conda.exe run -n work python make_ceiling_pa_dataset.py `
  --out-root ..\dataset_classroom_v2\generated `
  --num-train 10000 --num-valid 1000 --num-test 1000 `
  --seed 20260715 --split-seed 20260715
```

如果目标目录已有 wav，生成器会拒绝覆盖。只有确定旧数据可以删除时才增加
`--overwrite`。

### 12.5 正式微调

继续使用 `center=true`，从 VoiceBank 最佳权重初始化模型，但重新创建优化器：

```powershell
D:\Anaconda\Scripts\conda.exe run -n work python train_custom.py `
  --train-noisy ..\dataset_classroom_v2\generated\train\noisy `
  --train-clean ..\dataset_classroom_v2\generated\train\clean `
  --valid-noisy ..\dataset_classroom_v2\generated\valid\noisy `
  --valid-clean ..\dataset_classroom_v2\generated\valid\clean `
  --train-manifest ..\dataset_classroom_v2\generated\metadata\train.json `
  --valid-manifest ..\dataset_classroom_v2\generated\metadata\valid.json `
  --out-dir runs\classroom_v2 `
  --epochs 50 --batch-size 8 --lr 3e-4 `
  --scheduler warmup_cosine --warmup-epochs 3 `
  --warmup-start-lr 1e-6 --min-lr 1e-5 `
  --num-workers 4 `
  --seed 20260715 `
  --init-checkpoint runs\voicebank_serious_v1\checkpoints\best.tar
```

这里必须使用 `--init-checkpoint`，不能使用 VoiceBank 的 `--resume`。前者只加载
模型参数；后者还会恢复旧 optimizer、epoch 和训练历史。

### 12.6 正式评估

```powershell
D:\Anaconda\Scripts\conda.exe run -n work python evaluate_custom.py `
  --checkpoint runs\classroom_v2\checkpoints\best.tar `
  --noisy-dir ..\dataset_classroom_v2\generated\test\noisy `
  --clean-dir ..\dataset_classroom_v2\generated\test\clean `
  --manifest ..\dataset_classroom_v2\generated\metadata\test.json `
  --out-dir runs\classroom_v2\evaluation `
  --save-hardest 10 --save-worst 10
```

评估现在同时输出：

```text
hardest_samples:       输入 SI-SNR 最低的困难样本
worst_improvements:    增强后 SI-SNR 退化最多的失败样本
degraded_file_fraction: 增强后 SI-SNR 下降的文件比例
noise-only samples:    不计算无意义的 PESQ/STOI/SI-SNR，单独报告增强前后 RMS
mean_noise_attenuation_db: 纯噪声经过模型后的平均衰减量
```

### 12.7 已完成的 smoke test

2026-07-15 已生成 100/20/20 条 smoke 数据并完成一轮 GPU 训练和 20 条测试评估。
已验证：

```text
speaker、room_id 和 noise_file 在 split 之间重叠均为 0
所有输出均为 16 kHz 单声道 PCM_16
音频没有 NaN/Inf，峰值不超过约 0.98
早期反射目标、纯噪声零目标、HybridLoss 和反向传播可以共同运行
best checkpoint、hardest_samples 和 worst_improvements 均能生成
```

smoke 模型只训练了一个 epoch，指标没有模型质量意义，不能与正式模型比较。

### 12.8 接下来的执行顺序

严格按下面顺序进行，不跳过数据审计：

```text
Step 1: 生成正式 classroom_v2 数据（10000/1000/1000）。
Step 2: 审计 scene 比例、speaker/room/noise 隔离、SNR、RT60/DRR、
        speech_activity、采样率、声道、长度、峰值和 NaN/Inf。
Step 3: 数据审计通过后，从 VoiceBank best.tar 初始化，正式微调 50 epoch。
Step 4: 使用 valid loss 最低的 best.tar 评估未见房间测试集。
Step 5: 检查 hardest_samples 和 worst_improvements，进行人工听音。
Step 6: 加入未参与合成和训练的真实教室录音，作为最终验收集。
Step 7: 模型质量满足要求后，再建立 center=true 等效缓存的流式部署实验。
```

正式数据、训练目录和评估目录分别固定为：

```text
dataset:    D:\modeltraining\dataset_classroom_v2\generated
training:   D:\modeltraining\gtcrn\runs\classroom_v2
evaluation: D:\modeltraining\gtcrn\runs\classroom_v2\evaluation
```

### 12.9 正式 classroom_v2 数据状态（2026-07-15）

Step 1 和 Step 2 已完成。正式数据位于：

```text
D:\modeltraining\dataset_classroom_v2\generated
```

文件和清单数量：

```text
train: 10000 clean + 10000 noisy，manifest/metadata 各 10000 条
valid:  1000 clean +  1000 noisy，manifest/metadata 各 1000 条
test:   1000 clean +  1000 noisy，manifest/metadata 各 1000 条
```

第一次正式生成的数据审计发现，部分短语音在同一个 clean 文件内尝试 10 个
片段后仍低于 40% speech activity。生成器随后增加 `clean_file_attempts=10`：
片段不合格时继续更换 clean 文件。修正后重新生成了全部正式数据，所有含语音
样本的 `speech_activity >= 0.4`。

最终场景比例：

```text
                 train    valid    test
clean            10.05%    9.30%  10.30%
noise_no_reverb  10.09%   10.70%  10.40%
noise_only        4.71%    4.00%   5.50%
reverb_noise     60.16%   60.50%  60.10%
reverb_only      14.99%   15.50%  13.70%
```

最终审计结果：

```text
speaker overlap between every split pair: 0
room overlap between every split pair:    0
noise file overlap between split pairs:   0
exact RIR overlap between split pairs:    0

mean SNR: train 22.90 dB, valid 22.91 dB, test 22.36 dB
speech activity below 0.4: 0
all 24000 wav file sizes: 64044 bytes (2 s, mono PCM_16 at 16 kHz)
600 uniformly sampled wavs: correct shape/rate, no NaN/Inf
sampled maximum absolute peak: approximately 0.98
```

数据已通过正式训练前检查。下一状态为 Step 3：启动
`runs\classroom_v2` 的 50-epoch 微调。

正式训练前进行了 1000 train / 100 valid 的数据加载基准：

```text
batch_size=4, num_workers=4: 25.1 s/epoch, valid_loss=2.6224
batch_size=8, num_workers=4: 18.8 s/epoch, valid_loss=2.6269
```

最初的 `batch_size=4, num_workers=0` 正式试跑中，GPU 利用率约 30%，超过 10 分钟
仍未完成第一个 epoch，因此在产生 checkpoint 前终止。正式配置改为
`batch_size=8, num_workers=4`，保留 `lr=3e-4`；该配置通过了实际 GPU 基准。

### 12.10 正式训练暂停与用户终端续训（2026-07-15）

正式训练已按用户要求在完成 epoch 9 后停止，训练进程已退出。现有状态：

```text
last completed epoch: 9
last train loss:       2.53365065
last valid loss:       2.54838534
best epoch:            8
best valid loss:       2.50809863
resume checkpoint:     runs\classroom_v2\checkpoints\last.tar
best checkpoint:       runs\classroom_v2\checkpoints\best.tar
```

如果命令提示符显示为 `(work) D:\modeltraining\gtcrn>`，说明当前是已经激活
`work` 环境的 Windows CMD。直接复制下面完整的一行，不要加入反引号：

```bat
python train_custom.py --train-noisy ..\dataset_classroom_v2\generated\train\noisy --train-clean ..\dataset_classroom_v2\generated\train\clean --valid-noisy ..\dataset_classroom_v2\generated\valid\noisy --valid-clean ..\dataset_classroom_v2\generated\valid\clean --train-manifest ..\dataset_classroom_v2\generated\metadata\train.json --valid-manifest ..\dataset_classroom_v2\generated\metadata\valid.json --out-dir runs\classroom_v2 --epochs 50 --batch-size 8 --lr 3e-4 --scheduler warmup_cosine --warmup-epochs 3 --warmup-start-lr 1e-6 --min-lr 1e-5 --num-workers 4 --seed 20260715 --resume runs\classroom_v2\checkpoints\last.tar
```

CMD 不认识 PowerShell 的反引号续行符。只有提示符以 `PS` 开头、确定在
PowerShell 中时，才使用下面的多行版本。`--no-capture-output` 会让 conda 不再
缓存 Python 输出，可以直接看到每 20 个 step 和每个 epoch 的日志：

```powershell
cd D:\modeltraining\gtcrn

D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py `
  --train-noisy ..\dataset_classroom_v2\generated\train\noisy `
  --train-clean ..\dataset_classroom_v2\generated\train\clean `
  --valid-noisy ..\dataset_classroom_v2\generated\valid\noisy `
  --valid-clean ..\dataset_classroom_v2\generated\valid\clean `
  --train-manifest ..\dataset_classroom_v2\generated\metadata\train.json `
  --valid-manifest ..\dataset_classroom_v2\generated\metadata\valid.json `
  --out-dir runs\classroom_v2 `
  --epochs 50 --batch-size 8 --lr 3e-4 `
  --scheduler warmup_cosine --warmup-epochs 3 `
  --warmup-start-lr 1e-6 --min-lr 1e-5 `
  --num-workers 4 --seed 20260715 `
  --resume runs\classroom_v2\checkpoints\last.tar
```

续训会从 epoch 10 开始，并恢复 epoch 9 保存的模型、Adam optimizer 和最佳验证
损失，不会从头训练。`--epochs 50` 表示训练到总 epoch 50，不是再训练 50 轮。

### 12.11 classroom_v2 正式结果与交叉评估（2026-07-15）

正式训练已经完成 50 epoch，总训练时间约 2.50 小时。最佳 checkpoint 为：

```text
checkpoint: runs\classroom_v2\checkpoints\best.tar
best epoch: 48
best train loss: 2.32068384
best valid loss: 2.34743190
last epoch valid loss: 2.46291288
```

由于 epoch 50 的验证损失高于 epoch 48，正式推理和评估必须使用 `best.tar`，
不能使用 `last.tar`。

#### 12.11.1 classroom_v2 未见房间测试集

测试集包含 842 条非 clean 语音增强样本、103 条 clean 透明度样本和 55 条纯
噪声样本。不能把 clean 输入近乎无限高的输入 SI-SNR 与增强后有限 SI-SNR 相减，
否则少量 clean 样本会把总体 SI-SNR improvement 均值错误拉成负数。因此按场景
分别报告：

```text
非 clean 语音样本：842
mean SI-SNR improvement:   +1.3437 dB
median SI-SNR improvement: +1.3686 dB
improved fraction:          90.97%
mean PESQ improvement:      +0.3774
mean STOI improvement:      +0.0338

clean enhanced SI-SNR:       75.147 dB
clean PESQ change:           -0.0162
clean STOI change:           -0.0002

noise-only attenuation:      25.919 dB
```

正式输出：

```text
runs\classroom_v2\evaluation\summary.json
runs\classroom_v2\evaluation\metrics.csv
runs\classroom_v2\listening_samples
```

#### 12.11.2 四格交叉评估

两个 checkpoint 都使用同一评估脚本分别测试 VoiceBank 和 classroom 数据：

```text
model       test set    SI-SNR gain    PESQ gain    STOI gain
VoiceBank   VoiceBank     +9.3321       +0.6655      +0.0152
classroom   VoiceBank     +1.6911       +0.3620      +0.0042

VoiceBank   classroom     -0.2541       +0.1651      -0.0379
classroom   classroom     +1.3437       +0.3774      +0.0338
```

classroom 表中的 SI-SNR/PESQ/STOI 只统计 842 条非 clean 增强样本。补充对比：

```text
                              VoiceBank model   classroom model
classroom improved fraction       52.85%            90.97%
clean enhanced SI-SNR             24.998 dB         75.147 dB
clean PESQ change                 -0.4263           -0.0162
noise-only attenuation            17.482 dB         25.919 dB
```

结论：classroom 微调对目标域有明确且显著的收益，尤其改善了可懂度、clean 透明度
和纯噪声抑制；但 VoiceBank SI-SNR/PESQ/STOI 增益明显下降，说明只使用 classroom
数据微调造成了通用降噪能力遗忘。

下一版不能只扩大 classroom 合成数据。训练集应加入约 20%-30% VoiceBank 配对
样本作为 replay/rehearsal，并以分场景验证分数选择 checkpoint：

```text
70%-80% classroom：BUT RIR + PRESTO/PCAFETER + clean/noise-only 场景
20%-30% VoiceBank：原始通用噪声增强配对

checkpoint selection should monitor:
  classroom PESQ/STOI/SI-SNR
  VoiceBank PESQ/STOI/SI-SNR
  clean transparency
  noise-only attenuation
```

交叉评估输出：

```text
runs\cross_evaluation\voicebank_on_classroom
runs\cross_evaluation\classroom_on_voicebank
```

### 12.12 classroom_replay_v3 实验设计（2026-07-16）

`classroom_v2` 已证明场景微调有效，但也证明只用教室数据会产生明显的灾难性
遗忘。下一步只改变一个变量：在每个 epoch 中加入 VoiceBank replay，暂不改变
片段长度、STFT、数据总量和目标定义。

固定设置：

```text
初始化 checkpoint: runs\classroom_v2\checkpoints\best.tar
每轮样本总数:       10000
classroom_v2:        7500 (75%)
VoiceBank replay:    2500 (25%)
片段长度:            2 s
STFT:                16 kHz, win=160, hop=80, n_fft=256, center=true
batch size:          8
epochs:              30
max learning rate:   1e-4
checkpoint 选择:     0.75 * classroom valid loss + 0.25 * VoiceBank valid loss
```

每轮从两个训练池确定性地重新抽样；当源数据数量足够时不重复抽取。混合后的
10000 个索引再确定性打乱。训练 DataLoader 不使用 persistent worker，确保
`set_epoch()` 更新后的裁剪位置和 replay 日程真正传到 worker 进程。

PowerShell 正式训练命令：

```powershell
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py `
  --train-noisy ..\dataset_classroom_v2\generated\train\noisy `
  --train-clean ..\dataset_classroom_v2\generated\train\clean `
  --valid-noisy ..\dataset_classroom_v2\generated\valid\noisy `
  --valid-clean ..\dataset_classroom_v2\generated\valid\clean `
  --train-manifest ..\dataset_classroom_v2\generated\metadata\train.json `
  --valid-manifest ..\dataset_classroom_v2\generated\metadata\valid.json `
  --replay-train-noisy ..\dataset\train\noisy `
  --replay-train-clean ..\dataset\train\clean `
  --replay-train-manifest ..\dataset\splits\voicebank_serious\train.json `
  --replay-valid-noisy ..\dataset\train\noisy `
  --replay-valid-clean ..\dataset\train\clean `
  --replay-valid-manifest ..\dataset\splits\voicebank_serious\valid.json `
  --replay-fraction 0.25 `
  --epoch-size 10000 `
  --out-dir runs\classroom_replay_v3 `
  --epochs 30 --batch-size 8 --lr 1e-4 `
  --scheduler warmup_cosine --warmup-epochs 3 `
  --warmup-start-lr 1e-6 --min-lr 1e-5 `
  --num-workers 4 --seed 20260716 `
  --init-checkpoint runs\classroom_v2\checkpoints\best.tar
```

正式训练前先用小数据运行 1 epoch smoke test。训练完成后必须重复四格评估；
只有当 classroom 指标基本保持、VoiceBank 指标明显恢复时，才说明 replay 有效。
如果 VoiceBank 恢复不足，再比较 35% replay；如果教室域退化明显，再比较 15%。

#### 12.12.1 replay smoke test（2026-07-16）

使用 32 个训练项（24 classroom + 8 VoiceBank）、两边各 8 个验证文件、
`num_workers=2` 完成 1 epoch：

```text
device:                 cuda
train loss:             2.10715486
classroom valid loss:   1.69255197
VoiceBank valid loss:   2.87019908
weighted selection:     1.98696375
```

确定性日程单独检查结果为 7500/2500，两个源内均无重复；相同 seed 和 epoch
得到完全相同的日程，切换 epoch 后日程变化。smoke test 已验证 `metrics.csv`、
`training_curve.png`、`last.tar`、`best.tar` 和 checkpoint 内的三项验证指标均正常。
正式训练输出使用 `runs\classroom_replay_v3`，smoke 输出只保留在忽略目录
`runs\classroom_replay_v3_smoke`，不纳入 Git。

#### 12.12.2 正式训练手动接管（2026-07-16）

正式训练已完成前 2 个 epoch 后按用户要求终止，进程已退出，结果完整保存在：

```text
runs\classroom_replay_v3\metrics.csv
runs\classroom_replay_v3\checkpoints\last.tar
runs\classroom_replay_v3\checkpoints\best.tar
```

当前记录：

```text
epoch  lr        train     classroom valid  VoiceBank valid  selection
1      1e-6      2.5445    2.5898           3.0816           2.7128
2      5.05e-5   2.4056    2.4626           2.4528           2.4602
```

在 `(work) D:\modeltraining\gtcrn>` 的 Windows CMD 中，使用下面这一整行从
epoch 3 恢复。CMD 不支持 PowerShell 反引号续行，因此不能拆成之前那种多行命令：

```bat
python train_custom.py --train-noisy ..\dataset_classroom_v2\generated\train\noisy --train-clean ..\dataset_classroom_v2\generated\train\clean --valid-noisy ..\dataset_classroom_v2\generated\valid\noisy --valid-clean ..\dataset_classroom_v2\generated\valid\clean --train-manifest ..\dataset_classroom_v2\generated\metadata\train.json --valid-manifest ..\dataset_classroom_v2\generated\metadata\valid.json --replay-train-noisy ..\dataset\train\noisy --replay-train-clean ..\dataset\train\clean --replay-train-manifest ..\dataset\splits\voicebank_serious\train.json --replay-valid-noisy ..\dataset\train\noisy --replay-valid-clean ..\dataset\train\clean --replay-valid-manifest ..\dataset\splits\voicebank_serious\valid.json --replay-fraction 0.25 --epoch-size 10000 --out-dir runs\classroom_replay_v3 --epochs 30 --batch-size 8 --lr 1e-4 --scheduler warmup_cosine --warmup-epochs 3 --warmup-start-lr 1e-6 --min-lr 1e-5 --num-workers 4 --seed 20260716 --resume runs\classroom_replay_v3\checkpoints\last.tar
```

这里必须使用 `--resume`，因为需要连同第 2 轮的优化器、学习率进度和历史记录
继续；不要再使用 `--init-checkpoint`，否则会从 epoch 1 重新开始一个新优化器。

### 12.13 classroom_replay_v3 正式结果（2026-07-16）

训练完成 30 epoch。按 75% classroom valid loss + 25% VoiceBank valid loss
选择出的最佳 checkpoint 为：

```text
checkpoint:                runs\classroom_replay_v3\checkpoints\best.tar
best epoch:                11
30-epoch training time:    1.51 h
best classroom valid:      2.33713494
best VoiceBank valid:      2.05472725
best weighted selection:   2.26653302
```

训练 loss 最后约为 2.22，但正式使用 `best.tar`，不使用 epoch 30 的 `last.tar`。

#### 12.13.1 两域正式测试

教室测试使用未见房间的 1000 条 `classroom_v2/test`。语音指标只统计 842 条
非 clean、非 noise-only 场景；103 条 clean 和 55 条 noise-only 分开报告。

```text
model                 test set    SI-SNR gain  PESQ gain  STOI gain
VoiceBank baseline    VoiceBank      +9.3321     +0.6655    +0.0152
classroom_v2          VoiceBank      +1.6911     +0.3620    +0.0042
classroom_replay_v3   VoiceBank      +6.9098     +0.6020    +0.0070

VoiceBank baseline    classroom      -0.2541     +0.1651    -0.0379
classroom_v2          classroom      +1.3437     +0.3774    +0.0338
classroom_replay_v3   classroom      +1.2345     +0.3831    +0.0301
```

教室补充指标：

```text
metric                         classroom_v2   classroom_replay_v3
improved file fraction             90.97%             87.65%
noise-only attenuation             25.919 dB          24.091 dB
clean enhanced SI-SNR              75.147 dB          73.392 dB
clean PESQ change                  -0.0162            -0.0129
clean STOI change                  -0.0002            -0.0003
```

结论：25% replay 明显缓解了灾难性遗忘。VoiceBank SI-SNR 恢复了约 5.22 dB，
同时教室 SI-SNR 只下降约 0.11 dB，教室 PESQ 略有提高。`classroom_replay_v3`
应作为当前默认通用 checkpoint；如果部署永远只面对当前合成教室分布并且更重视
纯噪声衰减，可以保留 `classroom_v2` 作为专用对照模型。

正式评估输出：

```text
runs\classroom_replay_v3\evaluation_classroom_filtered
runs\classroom_replay_v3\evaluation_voicebank
```

`evaluate_custom.py` 已增加 `--metadata-csv`。存在场景元数据时，它会把 clean、
noise-only 和真实增强语音分开统计，并保证 `worst_improvements` 只从实际增强
场景选择，避免 clean passthrough 的极高输入 SI-SNR 污染均值和失败样本排序。

#### 12.13.2 试听

典型教室场景 A/B 目录：

```text
runs\classroom_replay_v3\listening_ab
```

共 6 组，每组按文件名中的编号依次试听：

```text
01_noisy              原始带噪/混响输入
02_v2_enhanced        只用教室数据微调的模型
03_v3_replay_enhanced 加入 25% VoiceBank replay 的模型
04_clean              训练目标参考
```

其中 `test_000946/test_000778` 为典型 reverb_noise，
`test_000380/test_000568` 为典型 reverb_only，
`test_000590/test_000971` 为典型 noise_no_reverb。

不要只听典型样本，还必须检查真实失败案例：

```text
runs\classroom_replay_v3\evaluation_classroom_filtered\worst_improvements
```

#### 12.13.3 下一步执行顺序

1. 冻结并备份当前 `best.tar`，停止继续调整合成训练参数。
2. 在目标教室采集一批真实录音，至少覆盖安静、风扇/设备、桌椅移动、远近说话、
   空教室纯噪声；记录麦克风、扬声器、距离和增益设置。
3. 对真实录音做 `noisy -> enhanced` 盲听。能获得同步干净参考的样本再计算
   SI-SNR/PESQ/STOI，普通现场讲话主要依靠盲听、语音可懂度和伪影记录。
4. 用实际部署的分块大小运行流式推理，对比离线输出，测量端到端延迟、块边界伪影
   和实时系数。当前模型使用 `center=true`，不能直接把离线结果当作实时结果。
5. 只有真实测试暴露出稳定问题后再训练下一版：通用噪声仍不足可比较 35% replay；
   教室效果下降明显可比较 15%；实时前视不可接受时再单独训练 `center=false` 版本。

下一阶段的首要工作是“真实教室验收 + 流式一致性测试”，不是立即扩大合成数据集。

## 13. classroom_v4 数据重构（2026-07-16）

在 `classroom_replay_v3` 已证明 25% replay 有效后，下一版扩展合成数据覆盖，
但不覆盖 `classroom_v2`。目标场景明确限制为普通平层教室：典型面积约
50-60 m²，最大约 100 m²，不考虑阶梯教室、礼堂和大型会议厅。

### 13.1 数据规模与不变项

```text
train: 20000 x 4 s = 22.22 h
valid:  1000 x 4 s =  1.11 h
test:   1000 x 4 s =  1.11 h
fs: 16 kHz mono
target: first 50 ms early reflections
scene probabilities: same as classroom_v2
SNR profile: quiet_classroom
```

4 秒语音由同一说话人的多条 VoiceBank 短句拼接，中间随机加入 80-300 ms
停顿，不使用循环重复或尾部补 2 秒零。train/valid/test 继续保持 speaker 隔离。

### 13.2 RIR 来源和教室尺寸限制

混响样本按以下权重选择来源：

```text
40% BUT ReverbDB real
20% RIRS_NOISES real
40% RIRS_NOISES simulated
```

BUT 根据 `env_meta.txt` 过滤，只保留面积 25-100 m²、层高不超过 4.5 m 的
普通房间。实际保留 C236、Q301、L207、L212；大型阶梯教室 D105、大型
lecture room E112、酒店大型会议厅、楼梯间和过小酒店客房全部排除。

RIRS real 只保留 office、meeting、RVB small/medium room 和 RWCP office。
明确标记为 largeroom、lecture、aula、stairway、booth、corridor 的 RIR 不使用。

simulated RIR 读取 `room_info`，严格限制：

```text
area:   40-100 m²
length: 5-15 m
width:  4-10 m
height: 2.5-4.2 m
```

筛选后得到 32 个 smallroom、3200 个 RIR；mediumroom 和 largeroom 均没有
满足本项目限制的房间。生成时进一步要求估计 RT60 在 0.15-1.5 s 内。
多通道真实 RIR 随机选单通道，不再直接平均多个通道。

所有来源均按物理 `room_id` 划分 train/valid/test，不按单个 RIR 文件随机拆分。

### 13.3 背景噪声与前景事件

MS-SNSD 从 Microsoft 官方仓库以 sparse checkout 下载，仅保留：

```text
dataset\MS-SNSD-sparse\noise_train: 128 files, 3.29 h
dataset\MS-SNSD-sparse\noise_test:   51 files, 0.66 h
```

连续背景来源权重：

```text
55% MS-SNSD semantic indoor categories
25% PRESTO/PCAFETER
10% RIRS isotropic noise
10% ESC-50 continuous categories
```

MS-SNSD 使用 AirConditioner、Office、Typing、CopyMachine、Hallway、
NeighborSpeaking、VacuumCleaner、WasherDryer 等室内类别；交通、车站、机场、
公园等室外类别不进入主体训练分布。官方 `noise_train` 只用于 train，官方
`noise_test` 再按文件拆为 valid/test。

ESC-50 使用 fold 1-3/4/5 对应 train/valid/test。明确语义的 door knock、door
creak、footsteps、keyboard、mouse click、clock alarm、clapping、coughing、
sneezing、water drops 等作为前景事件，以 10% 概率叠加在带噪语音场景。

前景事件不循环平铺，只在 4 秒片段中随机放置一次；按事件有效区间而不是整段
静音 RMS 缩放，SNR 主要为 18-30 dB，峰值不超过语音峰值的 0.8 倍。

### 13.4 smoke audit

500/100/100 smoke 数据和双份同 seed 复现检查均通过：

```text
train speakers: 25
train RIR pool: 2502
valid RIR pool: 1069
test RIR pool: 792
train background pool: 271
train ESC event pool: 288

speaker overlap: 0
room overlap: 0
background file overlap: 0
event file overlap: 0
minimum speech activity: 0.415
RT60 range after selection: 0.178-1.500 s
same-seed WAV hash equality: true
```

正式生成命令：

```powershell
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python make_classroom_v4_dataset.py `
  --out-root ..\dataset_classroom_v4\generated `
  --num-train 20000 --num-valid 1000 --num-test 1000 `
  --segment-seconds 4 --seed 20260717 --split-seed 20260717
```

### 13.5 正式 classroom_v4 状态

正式生成和 `audit_classroom_dataset.py` 审计已完成：

```text
dataset: D:\modeltraining\dataset_classroom_v4\generated
train: 20000 files, 22.22 h
valid:  1000 files,  1.11 h
test:   1000 files,  1.11 h
paired WAV files: 44000
WAV format: 16 kHz, mono, PCM16, 4 s
```

训练集实际分布：

```text
clean:             2031
reverb_only:       2993
reverb_noise:     11961
noise_no_reverb:   2001
noise_only:        1014

BUT RIR samples:        5414
RIRS real samples:      2271
RIRS simulated samples: 7269

MS-SNSD backgrounds:       8296
PRESTO/PCAFETER:            3747
RIRS isotropic backgrounds: 1466
ESC continuous backgrounds: 1467
ESC foreground events:      1389
```

训练集实际使用 27 个物理房间、2335 个 RIR、271 个背景噪声文件和 284 个
ESC 事件文件。背景 SNR 中位数 23.33 dB，事件 SNR 中位数 23.88 dB，RT60
中位数 0.377 s、范围 0.154-1.500 s。

最终审计：

```text
speaker train/valid/test overlap: 0
room overlap:                    0
background file overlap:         0
event file overlap:              0
all WAV file size:               128044 bytes
sampled WAV decoded:             2000
bad format/length:               0
NaN/Inf:                         0
peak over 0.981:                 0
silent noisy files:              0
audit passed:                    true
```

审计结果保存在：

```text
D:\modeltraining\dataset_classroom_v4\generated\metadata\audit.json
```

训练前试听目录：

```text
D:\modeltraining\dataset_classroom_v4\listening_samples
```

其中包含 clean、三种 RIR 来源、四种背景来源、door knock、footsteps、keyboard
typing 和 noise-only。正式训练前应先按 `manifest.csv` 听完这些输入/目标配对。

### 13.6 4 秒 VoiceBank replay

不能直接对原 VoiceBank 短语句设置 `segment_seconds=4`，否则大量片段会补零。
因此单独生成同说话人、paired noisy/clean 同步拼接的 replay 数据：

```text
dataset: D:\modeltraining\dataset_voicebank_replay_v4\generated
train: 10000 files, 11.11 h
valid:  1000 files,  1.11 h
segment: 4 s
train speakers: 25
valid speakers: 3
speaker overlap: 0
minimum speech activity: 0.4
```

所有 22000 个 paired WAV 均为 128044 字节；抽样 1000 个文件没有格式、NaN
或峰值错误。

### 13.7 classroom_v4 训练配置

每轮不遍历全部 20000 条 classroom 数据，而是确定性抽取：

```text
epoch size: 5000
classroom_v4: 3750 (75%)
VoiceBank replay v4: 1250 (25%)
segment: 4 s
batch size: 4
optimizer steps per epoch: 1250
audio per batch: 16 s
audio per epoch: 5.56 h
```

这与 v3 的 `10000 x 2 s, batch 8` 保持相同的每 batch 音频长度和每 epoch
更新次数。模型从 `classroom_replay_v3/best.tar` 初始化，新建优化器，最大 LR
降低到 `5e-5`，综合验证连续 6 轮不改善时自动停止。

32 条联合训练、两边各 8 条验证的 CUDA smoke 已通过：

```text
train loss:           0.4928
classroom valid:      1.9855
VoiceBank valid:      1.4899
selection loss:       1.8616
checkpoint fields:    valid
```

PowerShell 正式训练命令：

```powershell
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py `
  --train-noisy ..\dataset_classroom_v4\generated\train\noisy `
  --train-clean ..\dataset_classroom_v4\generated\train\clean `
  --valid-noisy ..\dataset_classroom_v4\generated\valid\noisy `
  --valid-clean ..\dataset_classroom_v4\generated\valid\clean `
  --train-manifest ..\dataset_classroom_v4\generated\metadata\train.json `
  --valid-manifest ..\dataset_classroom_v4\generated\metadata\valid.json `
  --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy `
  --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean `
  --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json `
  --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy `
  --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean `
  --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json `
  --replay-fraction 0.25 --epoch-size 5000 `
  --segment-seconds 4 `
  --out-dir runs\classroom_v4 `
  --epochs 30 --batch-size 4 --lr 5e-5 `
  --scheduler warmup_cosine --warmup-epochs 3 `
  --warmup-start-lr 1e-6 --min-lr 5e-6 `
  --num-workers 4 --seed 20260717 `
  --early-stopping-patience 6 `
  --init-checkpoint runs\classroom_replay_v3\checkpoints\best.tar
```

### 13.8 classroom_v4 正式训练与评估（2026-07-16）

训练在 epoch 18 触发 early stopping，累计 26.70 分钟。正式模型不是 epoch 18
的 `last.tar`，而是综合验证最好的 epoch 12：

```text
checkpoint:               runs\classroom_v4\checkpoints\best.tar
best epoch:               12
classroom valid loss:     0.83932108
VoiceBank replay loss:    2.06536637
weighted selection loss:  1.14583240

stopped epoch:            18
stopped selection loss:   1.22668256
epochs without improve:   6
```

epoch 18 的 train loss 继续下降到 1.1981，但验证结果没有超过 epoch 12，说明继续
拟合训练样本不会得到更好的双域模型。不能从 `last.tar` 继续训练来规避 early stopping。

#### 13.8.1 新 classroom_v4 未见房间测试

同一 1000 条 v4 test 上公平比较旧 v3 与新 v4：

```text
model   SI-SNR gain  PESQ gain  STOI gain  improved  noise attenuation
v3        +0.5586      +0.3545    +0.0146    73.34%       20.022 dB
v4        +0.8798      +0.4379    +0.0218    80.68%       21.007 dB
```

v4 在新的普通教室尺寸、多来源 RIR、MS-SNSD 和 ESC 分布上有一致提升，证明 4 秒
数据重构有效，不只是验证 loss 数字变小。

但是 clean passthrough 明显退化：

```text
model   clean enhanced SI-SNR  clean PESQ change  clean STOI change
v3             76.616 dB            -0.0080            -0.0001
v4             49.617 dB            -0.1217            -0.0037
```

49.6 dB SI-SNR 仍然很高，但 PESQ/STOI 表明 clean 输入的频谱细节被不必要地修改。

#### 13.8.2 旧 classroom_v2 测试

```text
model   SI-SNR gain  PESQ gain  STOI gain  improved  noise attenuation
v3        +1.2345      +0.3831    +0.0301    87.65%       24.091 dB
v4        +1.1458      +0.3814    +0.0317    84.32%       20.909 dB
```

旧域语音增强总体保持，STOI 略升，但 SI-SNR、改善比例和纯噪声衰减小幅下降。
旧域 clean PESQ change 从 v3 的 -0.0129 变为 v4 的 -0.1994，确认透明度问题
不是新测试集偶然现象。

#### 13.8.3 VoiceBank 官方测试

```text
model   SI-SNR gain  PESQ gain  STOI gain  improved
v3        +6.9098      +0.6020    +0.0070    99.88%
v4        +6.6594      +0.6478    +0.0093   100.00%
```

VoiceBank 没有明显灾难性遗忘。SI-SNR 小幅下降 0.25 dB，但 PESQ/STOI 提升，
说明 4 秒 replay 构造和 25% replay 比例总体有效。

正式评估目录：

```text
runs\classroom_v4\evaluation_v4_test
runs\classroom_v4\evaluation_v3_on_v4_test
runs\classroom_v4\evaluation_v2_test
runs\classroom_v4\evaluation_voicebank
```

v3/v4 同样本 A/B 试听：

```text
runs\classroom_v4\listening_ab

01_noisy
02_v3_enhanced
03_v4_enhanced
04_clean
```

目录同时包含 clean 和 noise-only，必须重点比较 `00_clean`，不能只听带噪样本。

#### 13.8.4 下一步：透明度修复，不继续扩大数据

当前不把 v4 直接替换为生产默认模型。v3 仍是 clean 透明度更安全的 checkpoint，
v4 是新复杂教室分布上更强的候选模型。

下一实验应从 `classroom_v4/best.tar` 做短程 identity repair，而不是重建 v5 数据：

```text
60% classroom_v4 non-clean enhancement scenes
15% classroom_v4 clean passthrough
25% VoiceBank replay v4

max epochs: 8
learning rate: 1e-5
early stopping patience: 3
```

训练脚本需要先增加 scene-aware sampling 和第三套 clean identity validation。
checkpoint selection 建议：

```text
0.60 * classroom enhancement valid loss
+ 0.25 * VoiceBank valid loss
+ 0.15 * clean identity valid loss
```

修复模型的接受门槛：

```text
v4 test SI-SNR gain >= 0.80 dB
v4 test PESQ gain >= 0.42
VoiceBank PESQ gain >= 0.62
clean PESQ change >= -0.03 on both classroom tests
old classroom noise attenuation >= 22 dB
```

达到这些门槛后再进行真实教室和流式 `center` 验收。未达到时保留 v3 作为默认，
不要因为 v4 在新合成测试上更强就忽略 clean 语音失真。

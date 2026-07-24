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

复现检查的规模说明（2026-07-17 补充）：同 seed 双份再生成检查对应根目录
`dataset_classroom_v4_repro_a/b`，规模为 train 5 / valid 3 / test 3，共 22 个
配对 WAV，哈希完全一致。该结论只覆盖这个小样本；正式 20000/1000/1000 数据
没有做过全量双份再生成验证。

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


## 14. clean 透传失真诊断（2026-07-17）

v4 的 clean passthrough 退化（clean PESQ change -0.12 ~ -0.20、clean enhanced
SI-SNR 49.6 dB，对比 v3 的约 76.6 dB）在修复前先做结构诊断，避免盲目调整
clean 比例。诊断结论直接决定 13.8.4 的修复手段选择。

### 14.1 方法

脚本：`gtcrn/diagnose_clean_passthrough.py`。对 v4 test 全部 102 条 clean 样本
（输入与目标同为干净语音）分别运行 v2/v3/v4 的 `best.tar`，测量：

```text
逐文件 SI-SNR / PESQ-WB / STOI（enhanced vs clean 参考）
等效传递函数 M_eff = stft(enhanced) / stft(input)
  （gtcrn.py 输出即内部 mask 与输入 STFT 的复乘；此处经过 ISTFT 波形
  往返后重新 STFT，是输出端等效传递函数，不是模型内部原始 mask 的
  精确还原；对频段/帧衰减诊断无影响）
0-1 kHz / 1-4 kHz / 4-8 kHz 频段能量增益
帧衰减 vs 帧能量（帧能量以文件峰值归一，考察峰值下 0-60 dB 范围）
瞬态帧（正向谱通量前 10%）与稳态帧的衰减差
失真与 speech_activity、输入响度的相关
```

输出：`runs/clean_diagnosis_v4_test/`（metrics.csv、analysis.json、plots/、
worst_spectrograms/ 最差 10 条语谱图）。

### 14.2 结果

102 条 clean 样本均值：

```text
model  SI-SNR(dB)  PESQ-WB  STOI      voiced 低能量帧 瞬态帧  稳态帧   4-8kHz
                                          帧衰减  衰减    衰减    衰减    频段增益
v2     77.0        4.635    0.99989  -0.002  -0.015  -0.001  -0.002  -0.002
v3     76.6        4.636    0.99991  -0.002  -0.008  -0.001  -0.003  -0.001
v4     49.6        4.522    0.99630  -0.047  -0.393  -0.023  -0.060  -0.034
（衰减/增益单位均为 dB；v4 min SI-SNR 20.4 dB，min PESQ 4.17，std 10.3 dB）
```

主要发现：

```text
1. 退化发生在 v3→v4。v2/v3 透明度几乎完美，排除“连续微调必然漂移”。
2. 主导失真是低能量帧压制：峰值下 20-50 dB 的帧平均衰减 -0.39 dB，
   为 voiced 帧（-0.05 dB）的约 8 倍；散点图显示个别安静帧衰减达 -13 dB，
   响帧基本不动。模型把噪声门式的“安静→抑制”行为泛化到了 clean 语音的
   停顿与弱音段。最差样本语谱图可见停顿/弱摩擦段被挖空。
3. 次要失真是 4-8 kHz 轻微多衰减（约为低频段的 1.5-2 倍），mask 相位误差
   也集中在 4-8 kHz（0.28° vs 其他频段 ≤0.05°）。
4. 瞬态帧衰减（-0.023 dB）反而小于稳态帧（-0.060 dB）：本次 clean 语音
   诊断不支持“ESC 事件导致模型过度压制攻击音”的假设。clean 测试不含
   真实 ESC 前景事件，不能排除 ESC 数据对训练行为的间接影响；但该假设
   未获本诊断支持，不作为修复设计依据。
5. 整体增益 ≈ 1（-0.025 dB），无全频段统一染色，mask identity 偏差小。
6. 文件级失真与输入响度无相关（r≈-0.01），与 speech_activity 仅弱相关
   （r≈-0.17）；v4 逐文件方差大，失真不是简单由响度或活动率解释。
```

### 14.3 首要怀疑原因（尚未隔离证实）

诊断已经证实的是“v4 压制 clean 输入的低能量帧”，但没有直接证明该行为
由哪个数据或训练因素造成。当前首要怀疑方向：

v4 的 4 秒拼接 clean 样本含 80-300 ms 合成停顿，且 speech_activity 下限
0.4 意味着最长约 60% 的低能量内容；noise_only 与安静背景场景可能教会
模型“低能量→接近零输出”。HybridLoss 的频谱 MSE 与 SI-SNR 项都按能量
加权，响亮帧主导梯度，安静帧上的 identity 约束信号很弱，模型在 clean
安静段学到压制的代价很低。v2/v3 的 2 秒自然语句停顿短，训练分布未暴露
该行为。

其他因素（clean 采样占比、noise_only 场景分布、checkpoint 选择方式、4 秒
片段长度本身）也可能与上述机制共同作用。隔离证实需要受控对比训练（例如
只改片段拼接方式、其余不变的实验），当前不做：无论根因是哪一个，14.4
的修复方向（提高 clean identity 约束强度并对低能量帧加权）都相同。
4 秒片段允许 60% 静音这一点 v2 同样存在（同为 0.4 下限），不能单独解释
为根因。

### 14.4 对 identity repair 的设计约束

诊断结果映射到修复手段：

```text
1. 主修复：按计划提高 clean identity 采样占比（13.8.4 的 15%），
   并采用硬门槛 checkpoint 选择。
2. 定向修复：clean identity 损失必须对低能量帧显式加权，否则标准
   HybridLoss 对安静帧约束过弱的问题依旧存在。候选实现：对 clean 样本
   增加帧能量比惩罚项 mean_t [10*log10(E_out/E_in)]^2，或在对数/压缩域
   计算 clean identity loss。该惩罚项只能作用于相对文件峰值 -50~-20 dB
   的有效低能量语音帧，并加 epsilon/clamp 保护：接近零能量的帧上
   log10(E_out/E_in) 数值不稳定，且不应强迫模型保留数值噪声底。
   诊断不支持 transient identity 约束（14.2 第 4 点）。
3. 硬门槛（在 valid 的 clean 子集上计算，不用 test）：
     clean PESQ change >= -0.03
     clean STOI change >= -0.001
     clean enhanced SI-SNR >= 65 dB，且不低于 v3 同验证集结果 10 dB 以上
   只有过门槛的 epoch 才参与 0.60/0.25/0.15 加权选择；无 epoch 过门槛则
   本次修复判定失败，退回 v3 为默认模型，不允许选“加权最好但 clean 仍
   失真”的 checkpoint。
4. v3 在 valid clean 子集上的基线使用 `diagnose_clean_passthrough.py`
   测量，结果记录在 14.5。训练命令通过
   `--clean-gate-reference-si-snr 77.5466` 使用该基线。
5. HybridLoss 数值随场景符号/量级变化大（SI-SNR 项为 -SI-SNR(dB)/10，
   clean 样本 loss 为大幅负值），硬门槛不能用“v3 的若干倍 loss”表达，
   只能用上述客观指标绝对阈值。
```

repair 训练完成后必须重跑完整矩阵：v4 test、v2 test、VoiceBank 官方测试、
两个 classroom test 的 clean transparency 与 noise-only attenuation，以及
v3/v4/repair 三模型同样本 A/B 试听，全部结果记录后再决定是否替换默认模型。

### 14.5 v4 valid clean 基线（2026-07-17）

使用 `dataset_classroom_v4/generated/metadata/valid.csv` 中全部 99 条 clean
样本，对 v3 和 v4 做同一协议测量：

```text
model  clean SI-SNR  PESQ change  STOI change  low-energy attenuation
v3       77.5466 dB    -0.00895     -0.000051       -0.00597 dB
v4       49.0848 dB    -0.13789     -0.001899       -0.39575 dB
```

结果目录：

```text
runs/clean_diagnosis_v4_valid/
```

因此 SI-SNR 硬门槛不是固定 65 dB，而是：

```text
max(65, v3 baseline - 10) = max(65, 77.5466 - 10) = 67.5466 dB
```

## 15. classroom_v4 identity repair v1（已完成，未通过）

### 15.1 已实现的训练行为

`train_custom.py` 已增加以下功能，同时保持旧训练命令兼容：

```text
scene-aware epoch schedule:
  60% classroom_v4 非 clean 场景（包含普通增强和 noise_only）
  15% classroom_v4 clean identity
  25% VoiceBank replay v4

clean identity loss:
  只作用于 clean identity 样本
  只计算目标峰值以下 -50~-20 dB 的有效低能量帧
  对帧能量增益 dB 的平方进行惩罚
  epsilon 防止除零，增益限制在 +/-20 dB
  默认本实验权重 0.1

validation:
  0.60 * classroom non-clean HybridLoss
  + 0.25 * VoiceBank replay HybridLoss
  + 0.15 * clean identity loss

hard gates:
  clean SI-SNR >= 67.5466 dB
  clean PESQ change >= -0.03
  clean STOI change >= -0.001
```

只有同时通过三条 hard gate 且 selection loss 改善的 epoch 才保存
`best.tar`。在首次出现合格 epoch 以前不会触发 early stopping；如果 8 个 epoch
全部不合格，目录中会有 `last.tar` 而没有 `best.tar`，该实验按失败处理并继续使用
v3。不能手工把未过门槛的 `last.tar` 当成最佳模型。

### 15.2 smoke test

已完成两种最小测试：

```text
scene-aware repair: 通过，clean gate=fail 时只保存 last.tar
legacy single-dataset training: 通过，旧命令仍可训练并保存 best.tar
200 条/epoch 的两轮方向性测试：clean SI-SNR 43.42 -> 44.89 dB，
  PESQ change -0.1842 -> -0.1834，STOI change -0.00266 -> -0.00239；
  三项都向改善方向移动，但小样本尚未通过 hard gate
py_compile: 通过
git diff --check: 通过
```

smoke test 只验证程序路径，样本数太少，其指标没有模型质量意义。

### 15.3 正式训练命令（Windows cmd 单行）

在 `D:\modeltraining\gtcrn>` 提示符后粘贴下面完整的一行。不要加入 PowerShell
反引号，也不要拆成多条命令：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v4\generated\train\noisy --train-clean ..\dataset_classroom_v4\generated\train\clean --valid-noisy ..\dataset_classroom_v4\generated\valid\noisy --valid-clean ..\dataset_classroom_v4\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v4\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v4\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.15 --epoch-size 10000 --out-dir runs\classroom_v4_identity_repair_v1 --segment-seconds 4 --epochs 8 --batch-size 8 --lr 1e-5 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --identity-energy-min-db -50 --identity-energy-max-db -20 --identity-gain-clamp-db 20 --primary-valid-weight 0.60 --replay-valid-weight 0.25 --clean-valid-weight 0.15 --clean-gate-min-si-snr 65 --clean-gate-reference-si-snr 77.5466 --clean-gate-max-si-snr-drop 10 --clean-gate-min-pesq-change -0.03 --clean-gate-min-stoi-change -0.001 --early-stopping-patience 3 --init-checkpoint runs\classroom_v4\checkpoints\best.tar --seed 20260717
```

训练过程中每个 epoch 会直接显示：classroom/replay/clean validation loss、
clean SI-SNR、PESQ change、STOI change 和 `clean_gate=pass/fail`。正式训练由用户
在可见终端中执行；Codex 不在后台代替启动。

### 15.4 正式结果（2026-07-17）

用户完成全部 8 个 epoch，总耗时 1701.9 秒（约 28.4 分钟）。没有 epoch 同时
通过三条 clean hard gate，因此训练目录只有 `last.tar`，没有 `best.tar`；按
15.1 的预定规则，本次实验判定失败，不能用 epoch 8 的 `last.tar` 替换 v3/v4。

```text
epoch  clean SI-SNR  PESQ change  STOI change  clean gate  selection loss
1        42.524 dB     -0.2061      -0.00289     fail          0.8033
2        42.066 dB     -0.2301      -0.00301     fail          0.8174
3        47.215 dB     -0.1476      -0.00173     fail          0.7099
4        54.981 dB     -0.0782      -0.00082     fail          0.5877
5        41.740 dB     -0.2318      -0.00276     fail          0.8318
6        40.369 dB     -0.2111      -0.00296     fail          0.8541
7        64.904 dB     -0.0438      -0.00041     fail          0.4136
8        44.738 dB     -0.1709      -0.00207     fail          0.7520
gate     67.547 dB     -0.0300      -0.00100
```

epoch 7 是最接近门槛且 selection loss 最低的候选：STOI 已通过，SI-SNR 还差
2.64 dB，PESQ change 还差 0.0138。说明 identity repair 方向有效，但 epoch 间
振荡过大。v1 脚本当时只保存过门槛的 `best.tar` 和最后的 `last.tar`，所以
epoch 7 权重已经丢失，无法补做完整测试矩阵。由于没有合格 checkpoint，本次
不运行 v4/v2/VoiceBank 正式测试和试听；对不合格的 epoch 8 做完整测试没有
模型选择意义。

### 15.5 BatchNorm 漂移诊断

GTCRN 含多个 `BatchNorm2d`。repair v1 虽然学习率只有 `1e-5`，训练模式仍会
每个 batch 更新 BatchNorm 的 running mean/variance。使用 v4 原始 checkpoint
做受控检查：学习率设为 0，只让 1 条训练样本经过一次训练模式，然后运行完整
99 条 clean-valid。

```text
condition                         SI-SNR     PESQ change  STOI change
v4 direct diagnosis              49.085 dB    -0.1379     -0.00190
1 sample, lr=0, BN updates        40.940 dB    -0.3308     -0.00478
1 sample, lr=0, frozen BN         49.085 dB    -0.1379     -0.00190
```

学习率为 0 时模型权重不会变化，指标仍下降约 8.1 dB；加 `frozen BN` 后与原始
v4 精确一致。因此 v1 大幅振荡的重要来源是 BatchNorm 统计量漂移，不应先通过
放宽门槛或扩大 clean 比例掩盖它。

## 16. classroom_v4 identity repair v2（下一次训练）

### 16.1 唯一主要变量

v2 保持 v1 的数据比例、identity loss、学习率、8 epoch 和 hard gate 不变，只
冻结 BatchNorm running statistics。BN 的 affine scale/bias 参数仍可反向传播；
冻结的只是运行均值和方差。这样既针对已证实的漂移原因，也保持实验可解释性。

训练脚本同时增加：

```text
--freeze-batchnorm:
  repair 训练时 BatchNorm 使用 v4 checkpoint 的 running statistics

--save-every-epoch:
  保存 checkpoints/epoch_001.tar ... epoch_008.tar

checkpoints/best_selection_candidate.tar:
  始终保存 selection loss 最低的候选，即使它没有通过 clean hard gate

clean_validation_curve.png:
  单独绘制 SI-SNR/PESQ/STOI 及三条 hard gate
```

`best_selection_candidate.tar` 只是诊断候选，不能替代通过门槛的 `best.tar`。

### 16.2 frozen-BN 方向性 smoke test

使用 200 条/epoch、20 条 clean-valid 做 2 epoch 小测试：

```text
epoch  clean SI-SNR  PESQ change  STOI change  gate
1        63.83 dB      -0.0219      -0.00005    fail (only SI-SNR)
2        72.34 dB      -0.0109      -0.00012    pass
```

该结果只证明梯度方向和 BN 冻结机制正确，不能代替 99 条 clean-valid 的正式
训练选择，更不能当成最终模型指标。

### 16.3 正式训练命令（Windows cmd 单行）

在 `D:\modeltraining\gtcrn>` 中执行下面完整的一行：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v4\generated\train\noisy --train-clean ..\dataset_classroom_v4\generated\train\clean --valid-noisy ..\dataset_classroom_v4\generated\valid\noisy --valid-clean ..\dataset_classroom_v4\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v4\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v4\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.15 --epoch-size 10000 --out-dir runs\classroom_v4_identity_repair_v2_frozen_bn --segment-seconds 4 --epochs 8 --batch-size 8 --lr 1e-5 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --identity-energy-min-db -50 --identity-energy-max-db -20 --identity-gain-clamp-db 20 --primary-valid-weight 0.60 --replay-valid-weight 0.25 --clean-valid-weight 0.15 --clean-gate-min-si-snr 65 --clean-gate-reference-si-snr 77.5466 --clean-gate-max-si-snr-drop 10 --clean-gate-min-pesq-change -0.03 --clean-gate-min-stoi-change -0.001 --early-stopping-patience 3 --freeze-batchnorm --save-every-epoch --init-checkpoint runs\classroom_v4\checkpoints\best.tar --seed 20260717
```

训练结束后先检查是否存在 `checkpoints/best.tar`。存在才进入完整测试矩阵；不
存在则 v2 仍失败，但所有 epoch 已保留，可以分析最接近门槛的具体 checkpoint，
不再重复 v1 丢失 epoch 7 的问题。

### 16.4 正式训练结果（2026-07-17）

v2 完成全部 8 个 epoch，总耗时 3929.3 秒（约 65.5 分钟）。冻结 BatchNorm 后
训练完全稳定，8 个 epoch 全部通过 clean-valid 三条 hard gate。按训练期
selection loss 选择的 `best.tar` 为 epoch 7：

```text
epoch  clean SI-SNR  PESQ change  STOI change  gate  selection loss
1        88.704 dB     -0.01145     -0.000127   pass      0.02725
2        92.464 dB     -0.00911     -0.000111   pass     -0.02625
3        95.506 dB     -0.01021     -0.000112   pass     -0.06637
4        96.995 dB     -0.00851     -0.000099   pass     -0.08699
5        95.154 dB     -0.01182     -0.000146   pass     -0.06280
6        94.768 dB     -0.00925     -0.000147   pass     -0.06613
7        98.466 dB     -0.01253     -0.000141   pass     -0.09870  <- best.tar
8        97.003 dB     -0.00994     -0.000151   pass     -0.09665
```

该结果确认 15.5 的判断：v1 的主要振荡来自 BatchNorm running statistics
漂移，`--freeze-batchnorm` 有效。但 clean-valid 通过只是进入测试矩阵的必要
条件，不等于最终模型已经被接受。

### 16.5 epoch 7 完整测试矩阵

新 classroom_v4 未见房间测试：

```text
model    SI-SNR gain  PESQ gain  STOI gain  improved  noise attenuation
v3         +0.5586      +0.3545    +0.0146    73.34%       20.022 dB
v4         +0.8798      +0.4379    +0.0218    80.68%       21.007 dB
repair     +0.9676      +0.3977    +0.0200    83.82%       23.669 dB

model    clean SI-SNR  clean PESQ change  clean STOI change
v3         76.616 dB        -0.0080            -0.0001
v4         49.617 dB        -0.1217            -0.0037
repair    101.568 dB        -0.0058            -0.0001
```

旧 classroom_v2 测试：

```text
model    SI-SNR gain  PESQ gain  STOI gain  improved  noise attenuation
v3         +1.2345      +0.3831    +0.0301    87.65%       24.091 dB
v4         +1.1458      +0.3814    +0.0317    84.32%       20.909 dB
repair     +1.1102      +0.3509    +0.0274    83.49%       23.244 dB

model    clean SI-SNR  clean PESQ change  clean STOI change
v3         73.392 dB        -0.0129            -0.0003
v4         42.366 dB        -0.1994            -0.0046
repair     82.854 dB        -0.0547            -0.0013
```

VoiceBank 官方测试：

```text
model    SI-SNR gain  PESQ gain  STOI gain  improved
v3         +6.9098      +0.6020    +0.0070    99.88%
v4         +6.6594      +0.6478    +0.0093   100.00%
repair     +6.3406      +0.5666    +0.0086    99.76%
```

预设接受门槛逐项判断：

```text
v4 test SI-SNR gain >= 0.80 dB:          pass (0.9676)
v4 test PESQ gain >= 0.42:               FAIL (0.3977)
VoiceBank PESQ gain >= 0.62:             FAIL (0.5666)
v4 test clean PESQ change >= -0.03:      pass (-0.0058)
v2 test clean PESQ change >= -0.03:      FAIL (-0.0547)
old classroom noise attenuation >=22 dB: pass (23.244)
```

因此 epoch 7 是透明度很好的诊断模型，但没有通过综合接受协议，不能替换 v3
作为默认模型。它提高了 SI-SNR、改善比例和 noise-only 衰减，却牺牲了 PESQ、
STOI 和 VoiceBank 泛化；不能只看 clean SI-SNR 约 100 dB 就宣布成功。

### 16.6 checkpoint 选择复核与消融

epoch 1 已经通过全部 clean-valid hard gate，理论上是改动最小的候选。额外运行
完整 v4 test 后：

```text
epoch 1: SI-SNR +0.9040, PESQ +0.3894, STOI +0.0186,
         noise attenuation 22.588 dB, clean PESQ change -0.0038
```

它同样未达到 v4 PESQ `>=0.42`，而且 PESQ 比 epoch 7 更低。因此失败不只是
“selection loss 错选了过度 repair 的 epoch 7”；模型一开始满足 identity 后，
增强 PESQ 已经发生回退。

把辅助 `identity_loss_weight` 从 0.1 降到 0.02 的 200 条/epoch 小消融中，clean
SI-SNR 仍按 `63.79 -> 71.59 -> 76.57 -> 80.42 dB` 快速上升，与 0.1 权重结果
接近。说明主要 identity 梯度来自 15% clean 场景上的基础 HybridLoss，而不只是
低能量辅助项。继续微调辅助权重不能解决多目标冲突；再试 clean 比例会进入缺少
真实数据约束的配方循环，当前停止。

### 16.7 试听目录

已在相同 12 组样本上生成五列对照：

```text
runs/classroom_v4_identity_repair_v2_frozen_bn/listening_ab/

01_noisy
02_v3_enhanced
03_v4_enhanced
04_repair_enhanced
05_clean
```

覆盖 clean、三类 RIR、MS-SNSD、PRESTO/PCAFETER、isotropic noise、ESC、敲门、
脚步、键盘和 noise-only。repair 可用于理解“透明度换增强质量”的听感，但由于
客观接受门槛失败，盲听偏好不能单独推翻完整矩阵。

### 16.8 项目决策与下一阶段

停止继续构造 synthetic identity repair v3。当前模型角色固定为：

```text
默认/保守模型: classroom_replay_v3/best.tar
  clean 透明度、旧 classroom、VoiceBank 综合最稳

新噪声分布候选: classroom_v4/best.tar
  v4 新教室 PESQ/STOI 最强，但 clean 透明度不合格

诊断模型: classroom_v4_identity_repair_v2_frozen_bn/best.tar
  clean 与 noise-only 很强，但综合 PESQ/泛化未通过，不用于默认部署
```

下一阶段不再训练合成模型，转入真实教室和流式验收：

```text
1. 用最终硬件和麦克风位置采集真实教室录音，保留未经处理的原始 wav。
2. 至少覆盖：安静讲话、空调/风扇、远距离讲话、多人活动、桌椅/脚步/键盘、
   noise-only；包含中文男声/女声和不同说话人。
3. 同一录音离线输出 v3/v4/repair，做不知道模型名称的 A/B/C 盲听。
4. 记录“语音自然度、噪声残留、混响、弱音/停顿是否被吞、事件声音是否突兀”。
5. 真实数据确认模型方向后，再做 center=true 离线结果与流式缓存输出的一致性、
   端到端延迟和扬声器泄漏/反馈测试。
```

真实验收以前不宣布 v4 或 repair 为生产模型，也不再根据合成分数继续调数据比例。

## 17. 无真实录音条件下的中文域补强（AISHELL-1）

### 17.1 前提变化

用户当前没有最终教室、麦克风安装位置和录音条件，无法执行 16.8 的真实录音
验收。因此项目不再声称已经覆盖真实教室分布；下一阶段改为补强当前最明确且可
验证的缺口：中文、更多说话人和低电平远距离语音。合成结果仍不能替代未来真实
硬件验收。

用户试听 AISHELL 原音后指出它并非绝对干声，带有轻微房间/混响感。该听感作为
数据定义约束：AISHELL 原音标记为 `native_room`，而不是 `dry_clean`。

### 17.2 批量解压工具与结果（2026-07-17）

新增脚本：`extract_aishell_archives.py`。源数据是 400 个按说话人拆分的
`Sxxxx.tar.gz`，脚本提供：

```text
tar member 路径穿越、链接和设备文件拒绝
每个归档解压后逐文件存在性与字节数校验
.extract_state/*.done.json 完成标记和 extract_log.jsonl
中断后重复运行时校验并跳过已完成包
--dry-run / --max-archives
--delete-archives-after-success（必须显式提供，默认绝不删除原包）
每包解压前剩余空间检查
```

正式命令：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python extract_aishell_archives.py --source-dir ..\dataset\data_aishell\wav --output-dir ..\dataset\data_aishell\wav_extracted
```

本机已完成全量解压且保留压缩包：

```text
archives:             400/400 verified
wav files:            141925
uncompressed size:    19.251 GiB
estimated duration:   179.38 hours
format sample audit:  1000/1000 = 16 kHz, mono, PCM16

split   speakers  wav files
train      340      120418
dev         40       14331
test        20        7176
```

官方 train/dev/test 已按说话人隔离，后续必须原样使用，禁止把同一 speaker 重新
随机分到不同集合。转写文件包含 141600 个 utterance id；325 个 WAV 无对应转写。
语音增强本身不使用文本，但生成清单默认排除这 325 条并记录原因。

当前下载包没有 `speaker.info` 或 gender metadata。400 位说话人能够提供明显高于
VoiceBank 25 人的多样性，也自然应包含男女声，但在获得官方 speaker metadata
前不能声称男女比例已被精确平衡；不使用基频自动猜性别作为正式标签。

### 17.3 中文 clean 零样本基线

从 AISHELL 官方 test 的 20 位未见说话人各取 5 条，共 100 条，分别测试原始
响度和统一缩放到 -25 dBFS 两种 clean passthrough。原始样本响度：

```text
mean -34.13 dBFS, range -43.28 to -24.88 dBFS
```

原始响度结果：

```text
model    clean SI-SNR  PESQ-WB  STOI    overall gain
v3          10.11 dB     2.363   0.906    -2.05 dB
v4           7.92 dB     2.407   0.882    -3.88 dB
repair       6.95 dB     2.178   0.827    -3.69 dB
```

统一到 -25 dBFS 后：

```text
model    clean SI-SNR  PESQ-WB  STOI    overall gain
v3          12.15 dB     2.559   0.940    -1.61 dB
v4          13.73 dB     3.064   0.952    -1.32 dB
repair      13.69 dB     2.864   0.948    -1.16 dB
```

响度归一化有明显帮助，但三个模型仍远未达到透明透传，证明缺口不只是低电平，
还包含中文说话人/发音结构、录音设备和 AISHELL 原生房间声。此前英语 clean
约 50-100 dB 的 SI-SNR 不能代表中文透明度。基线输出目录：

```text
runs/aishell_clean_baseline/
runs/aishell_clean_baseline_normalized/
```

### 17.4 classroom_v5_chinese 的目标定义

新数据不是把 AISHELL “去混响成干声”，因为不存在对应的真实干声参考：

```text
target = AISHELL 原始 native_room 语音（只做必要的响度缩放/裁剪）
input  = 同一语音 + 受控新增噪声/事件/额外短房间效应
```

基本原则：

```text
1. clean identity: input == target，必须保留原生轻微房间感。
2. additive noise: target 不变，只给 input 加空调/办公室/持续背景。
3. far speech: 语音分量与 target 同步降低电平；模型负责降噪，不负责把远距离
   人声恢复成不存在的近讲响度，最终扩声增益属于后级系统。
4. extra reverb: AISHELL 已有房间感，只使用普通教室短 RIR，建议 RT60
   0.15-0.55 s，并限制 wet mix；不再叠加强长尾 RIR。
5. events: 桌椅、脚步、键盘、敲门只出现一次，事件 SNR 以不遮住语音为主。
6. noise-only: target 为零，但比例保持小，避免重新教出过度噪声门。
```

计划先生成 `8000/800/800` 条 4 秒 paired WAV，控制磁盘占用，不直接复制全部
179 小时。建议场景分布的起始值：

```text
10% native_room identity（其中一半保留较低原始电平）
25% air-conditioner / office / fan-like additive noise
30% mild far-distance + short classroom RIR + quiet background
15% desk/chair/footstep/keyboard/door events
15% additive noise without extra RIR
 5% noise-only
```

背景 SNR 仍以 `18-32 dB` 为主，少量覆盖 `12-18 dB`。空调优先使用
MS-SNSD `AirConditioner`，风扇/设备持续声主要来自 OOFFICE；桌椅和键盘可用
MS-SNSD `SqueakyChair/Typing`，脚步/敲门沿用已筛选 ESC-50 事件。

### 17.5 执行顺序

下一步不直接训练，按以下顺序实现：

```text
Step 1: 新建 make_classroom_v5_chinese_dataset.py，读取官方 train/dev/test，
        排除无转写项并记录 native_room、speaker、source level、added RIR/noise/event。
Step 2: 生成 200/40/40 smoke 数据，试听 identity、低电平、短 RIR、事件和
        noise-only，确认没有把 AISHELL 原生房间声当作要删除的目标。
Step 3: 同 seed 生成两份 5/3/3 并做 WAV hash 复现检查。
Step 4: 生成 8000/800/800 正式中文数据并审计 speaker/RIR/noise 泄漏。
Step 5: 扩展训练验证为至少四域：中文 noisy、中文 raw/normalized clean、
        classroom_v4 non-clean、VoiceBank；所有域设置硬门槛并逐 epoch 保存。
Step 6: 先跑短 smoke fine-tune，再由用户终端运行正式训练。
Step 7: 完整评估中文 test、v4/v2 classroom、VoiceBank 和现有试听矩阵。
```

初始化 checkpoint 在实现 smoke 后再由对照决定：v3 在原始低电平中文上相对
更稳，v4 在归一化中文和新 classroom 上更好，当前不能只凭单项指标预先指定。

### 17.6 17.5 执行记录（2026-07-17）

执行前发现 `wav_extracted` 的 399/400 个说话人目录 ACL 损坏（全部拒绝访问，
仅 S0003 可读；非管理员 shell 无法 takeown/icacls）。经 UAC 提权执行
`takeown /R /D Y` + `icacls /reset /T /C /Q` 后 400/400 恢复可读，142731 个
文件处理成功、0 失败。损坏原因未定位，数据内容未受影响（损坏只发生在目录
ACL 层）。

**Step 1（完成）**：新增 `make_classroom_v5_chinese_dataset.py`。官方
train/dev/test 原样使用（340/40/20 说话人），325 条无转写 WAV 排除并记录到
`metadata/excluded_no_transcript.csv`。target = AISHELL native_room 原音
（只做电平缩放/裁剪，永不加 RIR）；identity 场景一半保留原始电平、一半缩放
到 -28~-22 dBFS；far_speech 场景语音分量与 target 同步降到 -40~-32 dBFS，
短 RIR 只作用于 input（RT60 限 0.15-0.55 s，wet mix 0.25-0.60，湿干混合后
按干声 RMS 归一）；背景 SNR 80% 取自 18-32 dB、20% 取自 12-18 dB；事件单次
放置、SNR 18-30 dB、峰值不超过语音 0.8 倍。噪声池：hvac = MS-SNSD
AirConditioner + OOFFICE；background = MS-SNSD 持续类 + OOFFICE + ESC
背景类；事件 = MS-SNSD SqueakyChair/Typing + ESC footsteps/door_wood_knock。
SqueakyChair 只存在于 MS-SNSD 官方 noise_train，因此只进 train 事件池。
修复了一个分裂泄漏 bug：AirConditioner 同属 hvac 与 background 两个语义池，
原先各池独立按不同 seed 拆 valid/test，导致同一 noise_test 文件经不同池同时
进入 valid 和 test；现改为按类别并集一次拆分、再按语义池过滤。

**Step 2（完成，试听待用户确认）**：`dataset_classroom_v5_chinese_smoke`
200/40/40 已生成，`audit_classroom_dataset.py` 通过；试听样本在
`dataset_classroom_v5_chinese_smoke/listening_samples/`（identity 原电平、
identity 归一化、far_speech 短 RIR 低电平、hvac、事件、noise_no_rir、
noise_only 共 13 对）。

**Step 3（完成）**：`dataset_classroom_v5_chinese_repro_a/b` 同 seed
5/3/3 双份再生成，22 个 WAV 哈希完全一致，metadata CSV/manifest 一致。
结论同样只覆盖该小样本。

**Step 4（完成）**：正式数据 `dataset_classroom_v5_chinese/generated`
8000/800/800（8.89/0.89/0.89 小时），审计通过：

```text
speaker/room/noise/event 跨 split 重叠: 全部 0
scene 分布: far_speech 2427, hvac 1996, noise_no_rir 1237,
           event 1205, identity 742, noise_only 393
RT60: 0.16-0.55 s (median 0.33)
背景 SNR: 12.0-32.0 dB (median 23.2)
事件 SNR: 18.0-30.0 dB
train 说话人: 340, RIR 文件: 1243, 噪声文件: 170, 事件文件: 69
语音活性: min 0.435, median 0.815
```

**Step 5（代码完成）**：`train_custom.py` 新增 `--validation-domains <json>`，
按域配置 noisy/clean 目录、scene 过滤、identity 标记、max_files、weight 和
硬门槛（min_si_snr_db / min_si_snr_change / min_pesq_change /
min_stoi_change / max_loss）。启用后替代原 primary/replay/clean 选择逻辑，
selection_loss 为域 loss 加权平均，best.tar 只在全部域过门槛且 selection
改善时保存；每域指标和 gate 结果逐 epoch 写入 metrics.csv，
`--save-every-epoch` 逐 epoch 存 checkpoint；旧训练命令行为不变。另新增
`--clean-scene-type`（默认 clean，v5 用 identity）和
`eval_validation_domains.py`（单 checkpoint 全域评估）。PESQ/STOI 对近静音
片段会抛 NoUtterancesError，域评估按文件跳过失败项。中文 clean 验证对由
`make_zh_clean_validation.py` 从 AISHELL dev 前 20 位说话人生成 60 对
identity（raw 原始电平 + normalized -25 dBFS），test 说话人保留给最终评估。

五域基线（`runs/*_v5domains_baseline.json`，正式 JSON，每域 64-128 文件）：

```text
checkpoint        zh_noisy   zh_clean_raw  zh_clean_norm  v4_nonclean  voicebank
v3 (si_snr)        8.83 dB     10.86 dB      13.31 dB       6.31 dB     10.64 dB
v4 (si_snr)        7.21 dB      7.92 dB      13.99 dB       6.55 dB     11.69 dB
```

v4 域 zh_noisy SI-SNR 相对 input 变化约 -10 dB：现有模型把低电平中文语音
当噪声压制，这正是 v5 数据要补的缺口。

**Step 6（smoke 完成，正式训练待用户终端）**：v4-init smoke fine-tune
（`runs/classroom_v5_chinese_smoke`，200/40/40 数据，2 epoch，lr 1e-5，
frozen BN，clean 0.15 / replay 0.25）管线验证通过，末 epoch：

```text
zh_noisy       8.63 dB (init 7.21)   gate fail（相对 input 仍 -9.23 dB）
zh_clean_raw  10.68 dB (init 7.92)   pass
zh_clean_norm 16.19 dB (init 13.99)  pass
v4_nonclean    6.68 dB (init 6.55)   pass（改善保留）
voicebank     10.97 dB (init 11.69)  pass（+5.92 dB 相对 input）
selection_loss 2.8365 -> 2.2146
```

（init 对照与正式训练命令见 17.7。）

### 17.7 init 对照结论与正式训练命令（2026-07-18）

v3-init 与 v4-init 在相同 smoke 数据、相同 recipe（2 epoch，lr 1e-5，
frozen BN，clean 0.15 / replay 0.25）下对照，末 epoch 域指标：

```text
domain         metric         v3-init   v4-init
zh_noisy       si_snr          10.59      8.63
               si_snr_change   -7.28     -9.23
zh_clean_raw   si_snr          14.17     10.68
zh_clean_norm  si_snr          16.43     16.19
v4_nonclean    si_snr           6.63      6.68
               si_snr_change   +1.76     +1.82
voicebank      si_snr           9.73     10.97
               si_snr_change   +4.68     +5.92
selection_loss                   2.55      2.21
```

结论：**正式训练以 v3（`runs\classroom_replay_v3\checkpoints\best.tar`）
初始化**。理由：v5 fine-tune 的目标域是中文，v3-init 三个中文域全胜
（noisy +1.96 dB、raw +3.48 dB、norm +0.25 dB），且归一化 clean 上 v4 的
基线优势在 2 个 smoke epoch 后即被反超；v4_nonclean 基本持平；v3-init 在
VoiceBank 上落后约 1.2 dB，但相对 input 仍保持 +4.68 dB 改善，由 0.25
replay 比例和 voicebank 硬门槛兜底。smoke 只验证管线与方向，其绝对指标
不代表正式模型质量。

正式训练硬门槛已写入 `validation_domains_v5.json`（锚定 v3 基线与 smoke
末 epoch 实测值，留有余量）：

```text
zh_noisy:      si_snr_change >= -8.0 dB,  stoi_change >= -0.07
zh_clean_raw:  si_snr >= 10.0 dB,         stoi_change >= -0.09
zh_clean_norm: si_snr >= 12.0 dB,         stoi_change >= -0.06
v4_nonclean:   si_snr_change >= +1.0 dB,  stoi_change >= +0.005
voicebank:     si_snr_change >= +3.0 dB,  stoi_change >= 0.0
```

best.tar 仅在五个域全部过门槛且加权 selection_loss 改善时保存；任一域
破门槛时保存 best_selection_candidate.tar 并继续训练。

正式训练命令（用户在终端运行）：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v5_chinese\generated\train\noisy --train-clean ..\dataset_classroom_v5_chinese\generated\train\clean --valid-noisy ..\dataset_classroom_v5_chinese\generated\valid\noisy --valid-clean ..\dataset_classroom_v5_chinese\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v5_chinese\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v5_chinese\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.15 --clean-scene-type identity --epoch-size 10000 --validation-domains validation_domains_v5.json --out-dir runs\classroom_v5_chinese --segment-seconds 4 --epochs 8 --batch-size 8 --lr 1e-5 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --freeze-batchnorm --save-every-epoch --early-stopping-patience 3 --init-checkpoint runs\classroom_replay_v3\checkpoints\best.tar --seed 20260717
```

Step 7 完整评估（中文 test、v4/v2 classroom、VoiceBank、试听矩阵）在正式
训练完成后执行。

### 17.8 试听校准：SNR 下调（2026-07-18）

用户试听 smoke 样本后确认：identity 直透正常（AISHELL 原生房间声保留），
但按 17.4 起始值（SNR 主 18-32 dB）生成的背景太安静，真实教室应更吵。
按用户选择改为：

```text
背景 SNR: 75% 取自 12-22 dB, 25% 取自 8-12 dB（原 80% 18-32 / 20% 12-18）
事件 SNR: 12-24 dB（原 18-30）
```

同 seed（20260717）重新生成 smoke 与正式 8000/800/800 数据，审计再次通过：
跨 split 零重叠，场景分布不变，背景 SNR 实测 8.0-22.0（median 15.3），
事件 SNR 12.0-24.0。试听样本已按新数据刷新到
`dataset_classroom_v5_chinese_smoke/listening_samples/`。
17.4 的 SNR 起始值以本节为准。zh_noisy 域门槛在新数据上重新校核（见 17.9）。

### 17.9 zh_noisy 门槛校核与正式训练启动（2026-07-18）

新 SNR 数据上 v3 基线：zh_noisy SI-SNR 7.94 dB、相对 input -4.56 dB、
stoi_change -0.0706（输入更吵后恶化幅度收窄）。其余四域数据未变，基线同
17.7。门槛最终定为：

```text
zh_noisy:      si_snr_change >= -5.5 dB,  stoi_change >= -0.09
zh_clean_raw:  si_snr >= 10.0 dB,         stoi_change >= -0.09
zh_clean_norm: si_snr >= 12.0 dB,         stoi_change >= -0.06
v4_nonclean:   si_snr_change >= +1.0 dB,  stoi_change >= +0.005
voicebank:     si_snr_change >= +3.0 dB,  stoi_change >= 0.0
```

用户授权由助手代为启动正式训练（命令同 17.7），输出
`runs\classroom_v5_chinese`。训练完成后执行 Step 7 完整评估。

### 17.10 Step 7 完整评估结果（2026-07-18）

**训练收敛**：epoch 6 后早停，`best.tar = epoch 3`（selection -2.4251），
五域门槛 epoch 2-6 全部通过。best epoch 五域 dev 指标：

| 域 | 指标 | v5 (epoch 3) | v3 基线 | v4 基线 |
|---|---|---|---|---|
| zh_noisy | SI-SNR dB（change） | 13.39（+0.32） | 7.95（-5.12） | 6.72（-6.35） |
| zh_clean_raw | SI-SNR dB | 94.95 | 10.86 | 7.92 |
| zh_clean_norm | SI-SNR dB | 82.04 | 13.31 | 13.99 |
| v4_nonclean | SI-SNR change dB | +1.39 | +1.54 | +1.75 |
| voicebank | SI-SNR change dB | +4.49 | +4.51 | +5.80 |

本表已在最终 8-22 dB SNR 数据上，以确定性随机抽取的同一批文件重算；旧版
表格曾误用 SNR 重构前的 v3/v4 基线，不能作为严格对照。中文三域从严重破坏
变为通过最低安全线；英文两域仍有正增益，但低于 v4。"过门槛"仅表示没有
灾难性退化，不等于产品质量认证。

**v5 test（800 条，分场景）**：评估目录 `runs\v5_eval\v5_test`。

| 场景 | n | input SI-SNR | enhanced | change | PESQ+ | STOI+ |
|---|---|---|---|---|---|---|
| identity | 75 | 139.43 | 80.86 | （透传） | -0.065 | -0.0025 |
| far_speech | 247 | 7.35 | 7.39 | +0.04 | +0.131 | +0.0001 |
| hvac_noise | 201 | 14.87 | 15.19 | +0.32 | +0.125 | +0.0019 |
| event | 113 | 19.00 | 18.54 | -0.46 | +0.070 | -0.0021 |
| noise_no_rir | 114 | 15.01 | 15.31 | +0.30 | +0.248 | +0.0070 |
| noise_only | 50 | — | — | 衰减 15.75 dB | — | — |

identity 中位数透传极好（native 104.7 dB / normalized 91.6 dB），但存在左尾：
native 最小 18.9 dB（-40 dBFS 轻语音）、normalized 最小 8.2 dB——残余风险点，
试听矩阵含 worst 样本供人工确认。noisy 场景 SI-SNR 基本持平（新 SNR 区间
input 已经较干净），PESQ 各场景平均为正。修正评估脚本把 `identity` 错算入
总体 speech 的问题后，675 条 non-identity speech 的总体 SI-SNR change 为
+0.082 dB，中位数 +0.003 dB，改善占比 77.63%，PESQ change +0.139，
STOI change +0.00142。

**回归对照（test 集）**：

| 测试集 | 模型 | SI-SNR+ | PESQ+ | STOI+ | 改善占比 | clean 透传 |
|---|---|---|---|---|---|---|
| v4 test | v5 | +0.63 | +0.234 | +0.0135 | 90.0% | 109.6 dB |
| v4 test | v4 | +0.88 | +0.438 | +0.0218 | 80.7% | 49.6 dB |
| v4 test | v3 | +0.56 | +0.355 | +0.0146 | 73.3% | 76.6 dB |
| v2 test | v5 | +0.87 | +0.260 | +0.0294 | 95.4% | 103.6 dB |
| v2 test | v4 | +1.15 | +0.381 | +0.0317 | 84.3% | 42.4 dB |
| VoiceBank | v5 | +6.12 | +0.427 | +0.0021 | 100% | — |
| VoiceBank | v4 | +6.66 | +0.648 | +0.0093 | 100% | — |
| VoiceBank | v2 | +1.69 | +0.362 | +0.0042 | 99.9% | — |

v5 在英文域增益略低于 v4，但改善占比显著更高（90.0%/95.4% vs
80.7%/84.3%），clean 透传从 50 dB 量级跃升到 100+ dB。

**AISHELL clean 对照（17.3 同口径，同 100 文件）**：评估目录
`runs\v5_eval\aishell_clean_raw` / `aishell_clean_norm`。

| 输入 | 模型 | SI-SNR dB | PESQ | STOI |
|---|---|---|---|---|
| raw | v3 | 10.11 | 2.363 | 0.906 |
| raw | v4 | 7.92 | 2.407 | 0.882 |
| raw | repair | 6.95 | 2.178 | 0.827 |
| raw | **v5** | **101.33** | **4.642** | **1.000** |
| -25 dBFS | v3 | 12.15 | 2.559 | 0.940 |
| -25 dBFS | v4 | 13.73 | 3.064 | 0.952 |
| -25 dBFS | repair | 13.69 | 2.864 | 0.948 |
| -25 dBFS | **v5** | **78.42** | **4.559** | **0.996** |

17.3 基线中中文 clean 被所有历史模型毁灭性破坏（SI-SNR 仅 7-14 dB、
PESQ 损失 1.6-2.4）；v5 raw 透传 101 dB、整体增益 -0.0001 dB，normalized
平均 78 dB。但 normalized 并非只有一个离群点：100 条中 SI-SNR <20 dB
有 6 条、<30 dB 有 14 条，P10 约 21.3 dB。因此更准确的结论是：大多数
中文 clean 已透明透传，左尾失真仍需保留为验收项，不能写成问题完全解决。

**试听矩阵**：`runs\v5_eval\listening\`（manifest
`runs\v5_eval\listening_manifest.json`），13 组 noisy/enhanced/clean 三元组，
覆盖 identity native/normalized 的 typical+worst、far_speech/hvac/event/
noise_no_rir 的 typical+低 SNR、noise_only typical。导出脚本
`export_listening_matrix.py`。

**结论与边界**：17.5 的主要目标（中文域补强、英文域不发生灾难性退化）在
合成指标上达成，但英文增强幅度低于 v4，中文 clean 仍有左尾离群。
但所有数据均为合成混合，指标达标不等于真实教室硬件场景验收通过；
上线前仍需真实设备录音的小规模人工验收。

### 17.11 评估口径修复与 runs 整理（2026-07-18）

审阅 17.10 后完成以下修复：

```text
1. evaluate_custom.py 将 identity 与 clean 一并作为 clean passthrough，排除出
   noisy speech 总体汇总；v5 test summary 已由错误的 -5.78 dB 修正为 +0.082 dB。
2. validation domain 不再截取 metadata/manifest 的前 N 条，改为按 sample_seed
   确定性随机抽取，避免文件顺序偏差。
3. PESQ 只跳过明确的 NoUtterancesError；其他异常立即失败。每域记录 items、
   pesq_items、pesq_skipped_no_utterances 和 stoi_items。
4. 增加 SI-SNR median、P10、<20 dB 和 <30 dB 比例。clean raw/norm 的 P10
   硬门槛分别设为 30/20 dB，防止平均值掩盖少量严重失真。
5. checkpoint selection 改为按域指标缩放后平均，不再直接平均数值尺度差异很大
   的 HybridLoss。旧训练的 best.tar 不被自动改写。
```

最终 SNR 数据、同一随机样本上的 v3/v4/v5 epoch 3 结果见 17.10 修正版表格。
所有域 PESQ/STOI 分母均已记录；v4_nonclean 的 PESQ 为 123/128，有 5 条因
NoUtterancesError 跳过，其余域均为全量。

按新选择公式复核旧 epoch 2-6 记录时 epoch 5 略优；在修复后的随机验证文件上：

```text
checkpoint       normalized selection   zh noisy+   raw P10   norm P10   v4+    VB+
epoch 3 best.tar       -2.4117             +0.324      45.94      27.76   +1.389 +4.489
epoch 5 candidate      -2.4329             +0.319      41.07      28.48   +1.411 +4.749
```

越低越好。epoch 5 在英文回放上稍强，但 normalized clean PESQ change 从
-0.062 降到 -0.084，因此暂不覆盖 `best.tar`；只有重跑完整 v5/v4/v2/VoiceBank/
AISHELL test 和试听矩阵后，才能决定是否把 epoch 5 提升为正式候选。

`runs/` 根目录的 6 个评估日志和 4 个历史基线 JSON 已整理到：

```text
runs/v5_eval/logs/
runs/v5_eval/validation_baselines/
```

旧 SNR 产物以 `old_snr18_32` 标记，仅保留追溯用途。仓库根目录新增
`RUNS_INDEX.md`，即使 `runs/` 被 Git 忽略，也能查到各产物的含义和位置。

### 17.12 epoch 5 候选的第一阶段完整测试与试听（2026-07-18）

暂不训练新模型。先比较原 `best.tar`（epoch 3）与按新归一化 selection 发现的
epoch 5。epoch 5 在完整 v5 test 的修正口径结果：

| 指标 | epoch 3 | epoch 5 |
|---|---:|---:|
| non-identity SI-SNR change | +0.082 | +0.105 |
| non-identity PESQ change | +0.139 | +0.167 |
| non-identity STOI change | +0.00142 | +0.00260 |
| SI-SNR 改善占比 | 77.63% | 74.96% |
| noise-only 衰减 | 15.75 dB | 17.36 dB |
| identity clean SI-SNR | 80.86 dB | 80.04 dB |
| identity clean PESQ change | -0.065 | -0.077 |
| identity clean STOI change | -0.00251 | -0.00329 |

分场景看，epoch 5 的 far speech、HVAC、event 的 PESQ/STOI 均优于 epoch 3；
event SI-SNR 退化也从 -0.464 缩小到 -0.411 dB。代价是 clean 透明度略降，
identity SI-SNR <30 dB 的文件从 8/75 增至 11/75。

AISHELL normalized clean 100 条直接对照进一步确认该取舍：

```text
metric                 epoch 3     epoch 5
mean SI-SNR             78.43       78.12 dB
minimum SI-SNR           9.04        8.79 dB
mean PESQ                4.559       4.533
mean STOI                0.99650     0.99518
low-energy attenuation  -0.381      -0.521 dB
```

epoch 5 对噪声与事件更积极，但也更容易削弱中文 clean 的低能量部分。它不是
可以仅凭平均分自动替换 epoch 3 的升级版本。当前停在人工试听门：

```text
教室场景 epoch 3: runs/v5_eval/listening/
教室场景 epoch 5: runs/v5_epoch5_eval/listening/

中文 normalized clean epoch 3:
runs/v5_epoch5_eval/aishell_clean_norm_compare/listening_epoch3/
中文 normalized clean epoch 5:
runs/v5_epoch5_eval/aishell_clean_norm_compare/listening_epoch5/
```

clean 对照选择了 epoch 5 相对 epoch 3 PESQ 退化最大的 5 条。每个文件夹内
同一 `clean_delta_XX` 的 `_enhanced.wav` 互相比较，并以 `_clean.wav` 为原音参考。
用户确认听感前不继续跑 v2/v4/VoiceBank 全矩阵，也不启动新训练。

### 17.13 用户试听结论与 v6 denoise repair smoke（2026-07-18）

用户对 epoch 3/5 的试听结论：

```text
event:       降噪能力很弱
far_speech:  降噪尚可
hvac_noise:  降噪尚可，但语音有轻微损伤感
其他场景:    噪声残留不够干净，整体不如早期英文训练模型
```

该反馈与指标一致。v5 的主要成功是中文 clean 透明度修复，non-identity test
SI-SNR 仅提高约 +0.08~+0.11 dB；训练时所有中文 noisy 被合并成一个验证域，
event 的负改善可以被 HVAC/background 抵消。仅换 epoch 或增加训练轮数不能解决
该问题。

下一版定义为 `v6 denoise repair`，先改变数据与验证结构，不改变 GTCRN 网络：

```text
scene distribution:
  identity       10%（正式训练仍由 scene-aware clean fraction 单独控制）
  hvac           25%
  far speech     20%（v5 为 30%，用户认为当前效果已经尚可）
  event          25%（v5 为 15%）
  noise no RIR   15%
  noise only      5%

SNR changes relative to v5:
  far speech:    不变
  HVAC:          -3 dB
  background:    -4 dB
  event:          6-18 dB（v5 为 12-24 dB）
  event peak:     speech peak 的 1.2 倍上限（v5 为 0.8）
```

`make_classroom_v5_chinese_dataset.py` 已参数化 scene fraction 和各场景 SNR
offset；默认值仍完全保持 v5 行为，只有显式提供 v6 参数时才生成更强噪声。

v6 smoke 已生成到 `dataset_classroom_v6_denoise_smoke`：

```text
train/valid/test: 300/60/60（0.333/0.067/0.067 小时）
train scenes: event 84, HVAC 68, far 56, background 51, identity 22,
              noise-only 19
background SNR: 4.08-21.43 dB，median 12.84
event SNR:      6.28-17.58 dB，median 12.75
speaker/room/noise/event split overlap: 全部 0
audio format/length/nonfinite/peak audit: 通过
```

试听样本位于：

```text
dataset_classroom_v6_denoise_smoke/listening_samples/
```

每组 `_noisy.wav` 是模型未来输入，`_clean.wav` 是目标。此阶段重点判断噪声是否
过强、是否仍能听清语音、event 是否符合教室桌椅/脚步/键盘/敲门的预期。用户
确认 smoke 数据分布以前，不生成正式数据、不启动训练。

### 17.14 为什么英文降噪更强、是否需要从头训练（2026-07-18）

用户确认 v6 第一版 smoke：event 清楚且强度正常，background 正是目标效果，
identity 正常；但 `hvac_low` 过强或源语音本身过小、发闷，far 的噪声又偏小。

英文效果较好不应归因为“英文比中文容易”，主要差别是训练条件：

```text
VoiceBank serious:
  10235 个严格配对的近讲 clean/noisy 文件
  clean 参考更接近真正干声
  2 秒 segment，lr 1e-3，最多 50 epoch

官方 GTCRN 预训练：
  DNS3 + VCTK-DEMAND，已经学习大量通用噪声抑制

中文 v5：
  AISHELL target 带原生房间感，部分源录音低电平或音色偏闷
  lr 1e-5，只训练到 epoch 6
  15% clean identity + clean 硬门槛，首先修复“不破坏中文”
  noisy 场景偏温和，并被合并成一个验证平均值
```

因此 v5 更像“中文透明度域适配”，不是一次完整的强降噪训练。当前 AISHELL
正式训练数据约 8.9 小时，也远不足以替代 DNS3/VCTK-DEMAND 预训练后从零学习。

**结论：不从头训练。** v6 应从 v5 epoch 3 初始化，只加载模型权重并新建
optimizer；这样保留通用降噪和已修好的中文透明度，再用更强且按场景验证的中文
数据做 denoise repair。从零训练只会增加数据量、训练时间和 clean 退化风险。

按用户试听生成第二版 smoke：`dataset_classroom_v6_denoise_smoke_b`。

```text
event:       保持 6-18 dB、peak ratio 1.2
background:  保持 v5 -4 dB offset（用户已确认）
HVAC:        从 -3 dB 放缓到 -2 dB offset
far speech:  从 0 改为 -3 dB offset
identity:    不变
```

smoke_b 仍为 300/60/60，审计通过，所有 split overlap 为 0；background SNR
整体 4.08-19.84 dB、中位数 12.04 dB。新试听位于：

```text
dataset_classroom_v6_denoise_smoke_b/listening_samples/
```

由于 event/background/identity 的源文件和参数未变，本轮只需复听 `hvac_*` 和
`far_*`。确认后才生成正式 12000/1200/1200 数据、建立 event/HVAC/far/
background 独立验证域，并提供用户终端训练命令。

### 17.15 far SNR 独立分布 smoke_c（2026-07-18）

用户复听 smoke_b 后认为 `far_typical_noisy.wav` 的噪声仍很小。该文件实际
SNR 约 17.2 dB；继续使用单一 offset 会把 low 样本一起推到过低 SNR，因此
生成器新增 far 专用双峰分布，而不是继续整体平移：

```text
75% far samples: 8-14 dB
25% far samples: 5-8 dB
```

默认不提供四个 far 专用边界时，生成器仍保持 v5 原行为。新参数必须四个一起
提供，并检查 `low_min <= low_max <= main_min <= main_max`。

第三版 `dataset_classroom_v6_denoise_smoke_c` 已生成 300/60/60 并通过审计；
split overlap 全部为 0。相同试听文件现在为：

```text
far_low:      6.98 dB（smoke_b 为 7.64 dB）
far_typical: 12.91 dB（smoke_b 为 17.19 dB）
```

只需试听：

```text
dataset_classroom_v6_denoise_smoke_c/listening_far/
```

本轮仍不生成正式数据、不训练模型；等待用户确认 far 强度。

### 17.16 far 噪声源修正 smoke_d（2026-07-18）

用户仍认为 smoke_c 的 `far_typical_noisy.wav` 几乎听不到噪声。检查发现该
样本继续使用 OOFFICE；数值 SNR 不能完全反映频谱上的主观可闻度，弱办公室
底噪即使放大仍可能听起来不明显。因此不再只降低 SNR，而是为 far 增加独立
噪声源权重：

```text
MS-SNSD continuous: 70%
OOFFICE:              0%
ESC background:      30%

far main SNR: 6-11 dB（75%）
far low SNR:  4-6 dB（25%）
```

生成器默认 far source 权重仍与 v5 background 相同，只有显式设置参数才使用
上述 v6 分布；权重必须非负且总和为 1。

`dataset_classroom_v6_denoise_smoke_d` 已生成 300/60/60 并通过全部审计。
试听文件：

```text
far_typical: MS-SNSD AirConditioner，SNR 10.43 dB
far_low:     ESC rain，SNR 5.57 dB
dataset_classroom_v6_denoise_smoke_d/listening_far/
```

用户确认这两类 far 输入后才锁定正式数据配置。

### 17.17 v6 正式数据、八域门槛与用户训练命令（2026-07-18）

用户确认 smoke_d 符合听感要求。使用完全相同的参数生成正式数据：

```text
dataset: dataset_classroom_v6_denoise/generated
train/valid/test: 12000/1200/1200
duration: 13.33/1.33/1.33 小时，共 16.0 小时
train speakers: 340
train rooms: 25
train RIR files: 1208
train noise/event files: 170/69
scene counts: event 3089, HVAC 2977, far 2360, background 1778,
              identity 1182, noise-only 614
background SNR: 4.00-20.00 dB，median 10.08
event SNR: 6.00-18.00 dB，median 11.98
RT60: 0.16-0.55 s，median 0.33
speaker/room/noise/event split overlap: 全部 0
500 WAV format/length/nonfinite/peak/silence audit: 通过
```

同参数 5/3/3 双份复现检查：除 `config.json` 中输出目录不同外，其余 29 个
文件（全部 WAV、CSV、manifest）SHA256 完全一致。

新增 `validation_domains_v6.json`，不再使用一个 zh_noisy 平均值，而是八域：

```text
v6_far / v6_hvac / v6_event / v6_background
zh_clean_raw / zh_clean_norm
v4_nonclean / voicebank
```

event selection weight 为 1.25，其余为 1.0。当前 v5 epoch 3 在正式 v6 valid
上的基线：

| domain | SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|
| v6_far | +0.308 | +0.109 | +0.00065 |
| v6_hvac | +0.696 | +0.141 | +0.00465 |
| v6_event | +0.023 | +0.017 | -0.00129 |
| v6_background | +1.097 | +0.188 | +0.00970 |
| v4_nonclean | +1.210 | +0.249 | +0.01317 |
| voicebank | +4.740 | +0.352 | +0.01290 |

clean raw/norm SI-SNR P10 为 45.94/27.76 dB。门槛锚定上述基线并要求 event
至少达到 SI-SNR +0.05、PESQ +0.02；因此初始化 v5 只有 event 域 FAIL，训练
必须实际改善 event 才能保存 `best.tar`。clean、v4、VoiceBank 均有硬保护线。

**不从头训练**：从 v5 epoch 3 加载模型权重，但不恢复 optimizer。使用
`lr=2e-5`、12 epoch、early stopping patience 4、frozen BN；每 epoch 12000
items = 60% v6 non-identity + 15% identity + 25% VoiceBank replay。

在 Windows CMD 的 `(work) D:\modeltraining\gtcrn>` 中粘贴下面完整一行：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v6_denoise\generated\train\noisy --train-clean ..\dataset_classroom_v6_denoise\generated\train\clean --valid-noisy ..\dataset_classroom_v6_denoise\generated\valid\noisy --valid-clean ..\dataset_classroom_v6_denoise\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v6_denoise\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v6_denoise\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.15 --clean-scene-type identity --epoch-size 12000 --validation-domains validation_domains_v6.json --out-dir runs\classroom_v6_denoise --segment-seconds 4 --epochs 12 --batch-size 8 --lr 2e-5 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --freeze-batchnorm --save-every-epoch --early-stopping-patience 4 --init-checkpoint runs\classroom_v5_chinese\checkpoints\best.tar --seed 20260718
```

已用同一命令将 `--epochs 12` 临时改为 `--epochs 0` 做只加载检查：CUDA、
STFT、所有数据路径、八域验证、replay 和 init checkpoint 均正常，未执行训练。

### 17.18 v6 训练结果、epoch 9 候选与完整回归（2026-07-18）

用户在本机终端完成 12 epoch，约 71 分钟。训练 loss 从 0.317 降到 -0.100，
event valid SI-SNR change 从 epoch 1 的 +0.426 提升到后期约 +1.2~+1.38 dB。

本轮**没有 `best.tar`**。八域硬门槛从未在同一 epoch 全部通过：epoch 1 仅
normalized clean PESQ 以 0.0014 的差距失败；epoch 3-9 主要是 v4_nonclean
在 SI-SNR/STOI 门槛附近或略低；epoch 10-12 又出现 clean P10 左尾下降。
`best_selection_candidate.tar` 是 epoch 12，仅代表 selection 数值最低，不代表
通过硬门槛，不推荐直接使用。

综合 valid 的 event、v4 和 clean P10，先选 epoch 7/9 做未见 v6 test 对照。
epoch 9 在 1200 条 v6 test 上全面优于 epoch 7，因此复制为：

```text
runs/classroom_v6_denoise/checkpoints/candidate_epoch_009.tar
```

复制后 SHA256 与 `epoch_009.tar` 一致。v6 test 总体：

| metric | v5 baseline | v6 epoch 9 |
|---|---:|---:|
| SI-SNR change | +0.395 | +1.011 |
| PESQ change | +0.148 | +0.321 |
| STOI change | +0.0021 | +0.0200 |
| improved fraction | 79.10% | 83.30% |
| noise-only attenuation | 14.40 dB | 18.85 dB |
| identity SI-SNR | 80.16 dB | 81.14 dB |

逐场景 v5 -> v6 epoch 9：

| scene | SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|
| far | +0.447 -> +1.260 | +0.140 -> +0.240 | +0.0007 -> +0.0302 |
| HVAC | +0.210 -> +0.520 | +0.188 -> +0.236 | +0.0065 -> +0.0083 |
| background | +1.097 -> +1.769 | +0.240 -> +0.353 | +0.0057 -> +0.0154 |
| event | +0.105 -> +0.845 | +0.056 -> +0.452 | -0.0037 -> +0.0266 |

用户指出的 event 弱抑制在合成 test 上得到明确修复。identity normalized P10
从 v5 的 21.20 提高到 27.05 dB，但两者仍各有 5 条 <20 dB，v6 最差样本
约 3.2 dB，仍是必须人工试听的左尾。

完整回归：

| test | SI-SNR+ | PESQ+ | STOI+ | clean SI-SNR |
|---|---:|---:|---:|---:|
| v6 test | +1.011 | +0.321 | +0.0200 | 81.14 |
| v4 test | +0.248 | +0.206 | +0.0047 | 87.57 |
| v2 test | +0.569 | +0.228 | +0.0189 | 83.05 |
| VoiceBank | +7.173 | +0.516 | -0.00036 | - |

相对 v5，目标 v6 与 VoiceBank 的 SI-SNR/PESQ 明显增强；旧 v4/v2 有回归，
VoiceBank STOI 轻微转负。AISHELL raw 仍接近透明（SI-SNR 97.90、PESQ 4.638、
STOI 0.99982）；normalized clean 平均 SI-SNR 69.12、PESQ 4.561、STOI 0.99531，
比 v5 更积极地压低低能量段，需试听判断是否可接受。

试听矩阵使用同一 13 组 noisy/clean，分别由 v5 和 v6 epoch 9 增强：

```text
runs/v6_eval/listening_v5_baseline/
runs/v6_eval/listening_epoch9/
```

重点听 `event_*`、`hvac_typical_failure`、`background_typical_residual`，以及
`identity_*_worst`。用户试听确认以前不继续训练。

方法边界：epoch 7/9 已使用 v6 test 做事后候选比较，因此该 test 不再是最终
无偏 holdout。若用户接受 epoch 9 听感，下一步冻结模型，不再调参；用新 seed
生成一套 confirmation 混合做一次最终确认，再决定是否提升为部署候选。

### 17.19 用户否决 v6 声场假设，重置为 continuous-only（2026-07-18）

用户明确否决 v5/v6 的两个核心假设：真实教室不会只有一次孤立键盘、敲门或
脚步声而没有持续底噪；v5/v6 的听感则落在“降噪太轻”与“降噪较强但损伤
人声”两个极端。用户要求不再使用这类孤立事件噪声，并对反复使用弱 OOFFICE
噪声表示不满。

该批评成立。v6 的 event scene 实际是 `speech + one transient`，没有 HVAC/fan
bed；这会要求模型压制与辅音瞬态相似的结构，增加人声损伤风险。与此同时，
弱 OOFFICE 与 identity 保护又推动模型保守，形成相互冲突的训练目标。

因此：

```text
v6 epoch 9 保留为实验 candidate，但用户听感未通过，不提升为正式模型。
不继续 v6 微调，不生成 confirmation set。
下一版从数据场景定义重新开始，而不是继续调 event 比例或 checkpoint。
```

生成器已增加可配置的 MS-SNSD category allowlist，以及 HVAC/background 独立
source weights；默认值保持历史 v5/v6 行为，显式 continuous-only 参数才启用
新分布。

新 smoke：`dataset_classroom_v7_continuous_smoke`，400/80/80，定义为：

```text
identity:       12%
near HVAC:      38%
far continuous:25%
near machine:   23%
noise-only:      2%
event:           0%

allowed noise categories: AirConditioner, CopyMachine
actual noise sources:      ms_ac, ms_continuous
OOFFICE: 0%; ESC: 0%; event files/categories: 0
near SNR: 75% 8-15 dB, 25% 4-8 dB
far SNR:  75% 6-11 dB, 25% 4-6 dB
```

审计通过：speaker/room/noise/event 跨 split 重叠全部为 0；1120 个 WAV 中抽查
100 个，格式、长度、峰值、静音、非有限数值全部正常。训练 metadata 明确显示
event_category 为空，noise_source 仅 `ms_ac/ms_continuous`。

试听目录：

```text
dataset_classroom_v7_continuous_smoke/listening_samples/
```

包含 AirConditioner/CopyMachine 各自的 near/far、typical/low，每组仅比较
`_noisy.wav` 输入和 `_clean.wav` 目标。用户确认该声场以前不训练任何新模型。

### 17.20 当前噪声生成方式、PRESTO/PCAFETER 位置与模型血缘校正（2026-07-18）

当前 continuous-only smoke 的噪声不是程序凭空合成的，流程是：

```text
1. 从 MS-SNSD 的 AirConditioner/CopyMachine 文件中随机裁 4 秒。
2. 按目标语音 RMS 和指定 SNR 缩放噪声：noise_rms = speech_rms / 10^(SNR/20)。
3. 与语音相加；far speech 还会先经过 BUT/RIRS 短 RIR，再叠加连续底噪。
4. 最后做峰值限制，并把 clean target 保持为未加新增噪声的 AISHELL native_room 语音。
```

当前 v7 smoke 的实际来源只有 `ms_ac` 和 `ms_continuous`，类别只有
AirConditioner、CopyMachine。

PRESTO/PCAFETER 检查结果：

```text
PRESTO:    16 个 mono/16 kHz 文件，约 1.33 小时
PCAFETER:  16 个 mono/16 kHz 文件，约 1.33 小时
```

它们适合模拟持续的学生低声讨论，但不应单独成为 noisy input，否则模型会把
“没有空调/风扇/设备底噪的纯讨论声”当作完整教室分布。下一版建议：

```text
base bed:   AirConditioner/CopyMachine，所有带噪语音都有
murmur bed: PRESTO/PCAFETER，只在约 25-35% 的带噪场景叠加
murmur SNR: 约 15-24 dB，低声、不可清晰辨认内容
```

这样才能表示“持续机械底噪 + 偶尔学生窸窣讨论”的真实教室，而不是孤立事件。
已生成 layered-murmur smoke：
`D:\modeltraining\dataset_classroom_v7_murmur_smoke\generated`。
共 400/80/80 个 train/valid/test 样本，其中 30% 的非 identity、非 noise-only
样本叠加 PRESTO/PCAFETER；PRESTO 与 PCAFETER 各贡献 70 个样本。用户试听确认
学生讨论电平和持续底噪符合目标，因此可以进入正式数据生成。

**模型血缘校正**：此前文档把官方 DNS3/VCTK 权重与当前实验链混在一起，现明确
纠正：

```text
官方 checkpoint（存在但未用于当前自定义链）：
  checkpoints/model_trained_on_dns3.tar
  checkpoints/model_trained_on_vctk.tar

实际当前链：
  VoiceBank serious v1：自定义 train_custom.py，从随机初始化开始
  classroom_v2：从 VoiceBank serious best.tar 初始化
  classroom_replay_v3：从 classroom_v2 best.tar 初始化
  v5 Chinese：从 classroom_replay_v3 best.tar 初始化
  v6 candidate epoch 9：从 v5 best.tar 初始化，继续微调
```

官方 VCTK/DNS3 checkpoint 的 STFT 为 `n_fft=512`；当前自定义链使用 `n_fft=256`。
实测官方权重加载到 `GTCRN(nfft=256)` 有 2 个 shape mismatch，不能直接当作当前
模型初始化；加载到 `n_fft=512` 才形状一致。未来若比较官方初始化，必须单独做
`n_fft=512` 对照实验，不能把它和现有 v5/v6 数字直接混称。

### 17.21 v7 continuous + student murmur 正式数据（2026-07-18）

用户确认 layered-murmur smoke 听感符合普通 50-100 平方米教室目标，正式数据
固定为：

```text
base noise: MS-SNSD AirConditioner / CopyMachine
murmur layer: PRESTO / PCAFETER
event scenes: 0%
OOFFICE / ESC-50: 0%（不进入主体噪声）
train/valid/test: 8000/800/800（约 8.89/0.89/0.89 小时；noisy/clean 是配对文件，
不重复计入时长）
segment: 4 s, fs=16 kHz, center=true
murmur: 30% of non-identity/non-noise-only; 75% at 15-24 dB,
        25% at 10-15 dB
RIR: BUT ReverbDB and RIRS small/medium rooms, configured area 25-100 m2
```

生成顺序：先生成 continuous-only 数据，再运行
`add_student_murmur_bed.py` 叠加讨论层。正式输出为
`D:\modeltraining\dataset_classroom_v7\generated`，实际统计为：

```text
train/valid/test: 8000/800/800
student murmur: 2467（PRESTO 1224，PCAFETER 1243）
scene distribution (train): far 1949, HVAC 3012, identity 999,
                            background 1878, noise-only 162
RIR: 25 个房间，面积约束 25-100 m2；event=0
speaker/room/noise/event 跨 split overlap: 0
audio audit: 19200 WAV，16 kHz/4 s，格式、长度、非有限值、静音、峰值均通过
```

PRESTO/PCAFETER 的 16 个同步通道按录音时间切分，而不是按通道文件切分：
train 使用 0%-70%，valid 使用 70%-85%，test 使用 85%-100%，避免同一时刻的
学生讨论声泄漏到不同 split。新增的 `student_murmur_start_sample` 和
`student_murmur_source` 会记录该切分位置和来源。

v3 与 v5 初始化对照：在 v7 valid 的快速对照中，v3 的非 identity 语音 SI-SNR
change 为 -4.50 dB，而 v5 在同一批 256 条样本上为 +0.41 dB，且 v5 的 clean
PESQ/STOI 退化接近 0。因此 v7 正式训练从
`runs\classroom_v5_chinese\checkpoints\best.tar` 初始化，并保留 25% VoiceBank
replay；不使用 v6 candidate。

验证域配置为 `validation_domains_v7.json`，额外区分持续底噪和 murmur 层，并
对中文 clean raw/normalized 设置硬保护线。训练管线加载检查已通过（epochs=0），
尚未执行正式训练。用户手动训练命令：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v7\generated\train\noisy --train-clean ..\dataset_classroom_v7\generated\train\clean --valid-noisy ..\dataset_classroom_v7\generated\valid\noisy --valid-clean ..\dataset_classroom_v7\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v7\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v7\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.12 --clean-scene-type identity --epoch-size 12000 --validation-domains validation_domains_v7.json --out-dir runs\classroom_v7 --segment-seconds 4 --epochs 20 --batch-size 8 --lr 2e-5 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --freeze-batchnorm --save-every-epoch --early-stopping-patience 5 --init-checkpoint runs\classroom_v5_chinese\checkpoints\best.tar --seed 20260722 --overwrite-run
```

### 17.22 v7 正式训练、test 对照与试听（2026-07-18）

用户在本机终端完成全部 20 epoch，未触发 early stopping，总训练时间约
108.7 分钟。六个验证域从 epoch 1 起全部通过硬门槛。综合选择最优为 epoch 15：

```text
runs/classroom_v7/checkpoints/best.tar
checkpoint epoch: 15
selection loss: -2.43928227
```

epoch 20 的 continuous/murmur/far SI-SNR change 略高，但 epoch 15 的综合选择
分数与 clean 透明度更稳，因此正式评估使用 `best.tar`，不用 `last.tar`。

在未参与 checkpoint 选择的 800 条 v7 test 上，与 v5 epoch 3 对照：

| metric | v5 | v7 best |
|---|---:|---:|
| SI-SNR change | +0.020 dB | +0.714 dB |
| PESQ change | +0.253 | +0.395 |
| STOI change | +0.0107 | +0.0244 |
| improved fraction | 69.44% | 83.48% |
| noise-only attenuation | 17.39 dB | 16.95 dB |
| clean enhanced SI-SNR | 85.87 dB | 92.39 dB |
| clean PESQ change | -0.0280 | -0.0199 |
| clean STOI change | -0.00097 | -0.00075 |

v7 相对 v5 的目标域提升成立，同时 clean 透明度没有被牺牲。noise-only 衰减
少 0.44 dB，但不能用这一个指标否定模型，因为含语音样本的 SI-SNR/PESQ/STOI
和改善比例均更好。

v7 test 分场景：

| subset | files | SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|---:|
| continuous without murmur | 481 | +0.598 | +0.374 | +0.0206 |
| student murmur | 203 | +0.988 | +0.444 | +0.0332 |
| HVAC | 312 | +0.780 | +0.433 | +0.0252 |
| machine/no-RIR | 172 | +1.069 | +0.454 | +0.0241 |
| far speech | 200 | +0.305 | +0.285 | +0.0233 |

far speech 仍是最弱场景，不应只看总平均值。最差 SI-SNR 样本
`test_000480.wav` 为 -10.77 dB，但 PESQ 仍提高 +0.141，说明 SI-SNR 与听感可能
冲突，必须人工试听。

试听矩阵：

```text
runs/v7_eval/listening/01_v5/
runs/v7_eval/listening/02_v7_best/
```

两边文件名前缀一致，每组包含 noisy/enhanced/clean。01-04 为 HVAC、机器、
PRESTO、PCAFETER；05-06 为 far；07 为 identity；08 为最差 SI-SNR 样本。

下一步先完成人工 A/B：重点确认 03/04 学生讨论、05/06 远讲、07 clean 透明度、
08 是否真实听感退化。若整体通过，停止继续调训练数据，运行 v4/v2/VoiceBank
完整回归和流式 center 一致性测试，再决定是否提升为部署候选；若不通过，只按
具体失败场景诊断，不立即从头训练。

### 17.23 v7 听感失败原因与 v7.1 三档 SNR smoke（2026-07-18）

用户试听认为 v7 整体降噪偏弱，而少数降噪明显样本又损伤人声。复核正式数据
`config.json` 后发现，v7 正式集没有使用此前试听确认的 continuous smoke SNR：

```text
计划/试听 smoke near: 75% 8-15 dB, 25% 4-8 dB
计划/试听 smoke far:  75% 6-11 dB, 25% 4-6 dB

v7 正式实际 near/far: 75% 12-22 dB, 25% 8-12 dB
训练 metadata 实际范围: 8.00-22.00 dB, median 15.42 dB
```

原因是生成正式数据的命令没有显式传递 SNR 参数，回落到了 v5 生成器默认值；
far 专用 SNR 参数也未传，因此 far 同样使用较安静的全局分布。这解释了模型为何
大部分时候降噪偏弱。旧审计只确认数值合法，没有比较 smoke 与正式 config，未能
发现该偏差。v7 保留为失败实验，不作为部署候选。

同时，v7 test 中高输入 SNR 样本存在明显左尾，例如 `test_000480.wav` 输入
SI-SNR 约 21.48 dB，增强后 SI-SNR change 为 -10.77 dB。这说明只增加低 SNR
样本会推动模型更激进，不能解决“弱降噪与人声损伤”并存的问题。

生成器已加入向后兼容的第三档 high SNR；不提供新参数时旧命令行为不变。
v7.1 固定分布为：

```text
near low:  20%, 4-8 dB
near main: 60%, 8-15 dB
near high: 20%, 15-24 dB

far low:   25%, 4-6 dB
far main:  60%, 6-11 dB
far high:  15%, 11-18 dB
```

训练验证 metadata 过滤也增加 `numeric_filters`，后续可以分别建立 low-SNR
降噪域和 high-SNR do-no-harm 域，不再只看场景平均值。

v7.1 smoke 位于：

```text
D:\modeltraining\dataset_classroom_v7_1_smoke\generated
```

规模 400/80/80，叠加 146 条 PRESTO/PCAFETER。审计结果：

```text
background SNR: 4.08-23.53 dB, median 10.31 dB
HVAC train low/main/high: 37/78/27
machine train low/main/high: 21/48/20
far train low/main/high: 22/58/21
speaker/room/noise/event split overlap: 0
1120 WAV format/length/nonfinite/peak/silence audit: pass
numeric validation filter <=8 dB / >=15 dB: verified
```

输入试听目录：

```text
D:\modeltraining\dataset_classroom_v7_1_smoke\listening_samples
```

01-03 为 HVAC low/main/high，04-06 为机器 low/main/high，07-09 为 far
low/main/high，10 为讨论声，11 为 identity。当前只确认数据强度；用户确认以前
不生成正式 v7.1、不启动任何训练。

### 17.24 用户反馈：三档整体下移，smoke_b（2026-07-18）

用户试听 v7.1 smoke 全部 11 组后反馈：**所有文件都不吵，只有 01/04/07
（即 low 档）才是其认为正常的教室噪声**。对应实测 SNR：01 hvac 6.54 dB、
04 machine 6.21 dB、07 far 5.13 dB。结论：原 main 档（near 8-15 dB）就已经
偏干净，三档需整体下移，让"用户认定的正常强度"成为 main 档。

新三档分布（生成器参数不变、仅数值下移，seed 与 v7.1 smoke 相同，因此底层
语音/房间/噪声文件与上一轮一致，只有噪声缩放不同，可直接 A/B）：

```text
near low:  20%, 1-4 dB     （原 4-8）
near main: 60%, 4-8 dB     （原 8-15，= 用户认定的正常档）
near high: 20%, 8-14 dB    （原 15-24；保留安静教室 do-no-harm 保护）

far low:   25%, 2-4 dB     （原 4-6）
far main: 60%, 4-6 dB      （原 6-11）
far high: 15%, 6-10 dB     （原 11-18）
```

high 档不再追求"更干净的上限"，只保留到原 main 档下沿附近，原因是 17.23 的
左尾教训：高输入 SNR 样本仍需要训练覆盖，否则模型在安静场景容易过度抑制。

smoke_b 已生成并通过全部审计：

```text
base:     dataset_classroom_v7_1_continuous_smoke_b/generated
layered:  dataset_classroom_v7_1_smoke_b/generated（400/80/80）
murmur:   146 条（PCAFETER 82，PRESTO 64），参数与 v7.1 smoke 相同
train low/main/high: hvac 37/78/27, machine 21/48/20, far 22/58/21
background SNR: near 1.1-13.7 dB（median 5.9），far 2.1-9.7 dB（median 5.1）
speaker/room/noise/event split overlap: 0
1120 WAV format/length/nonfinite/peak/silence audit: pass
```

试听目录（结构与上轮相同，01-11；多数样本底层内容与上轮同文件、仅噪声更
响，便于直接对比）：

```text
D:\modeltraining\dataset_classroom_v7_1_smoke_b\listening_samples
```

01 hvac 2.9 dB / 02 hvac 6.4 / 03 hvac 10.4 / 04 machine 2.7 / 05 machine
6.7 / 06 machine 10.7 / 07 far 3.1 / 08 far 5.0 / 09 far 6.7 / 10 murmur
（far 底噪 6.0 dB + PRESTO 17.0 dB）/ 11 identity。用户确认强度以前不生成
正式数据、不训练。若 main 档（02/05/08）被认可，正式 v7.1 即按此分布生成；
注意 01/04/07 现在比上轮更响一档，需确认 low 档没有强到掩盖语音。

### 17.25 v7.1 正式数据、SNR 分档验证域与训练命令（2026-07-18）

用户确认 smoke_b 全部 11 组"比较符合"。按完全相同的三档分布生成正式数据：

```text
base:    dataset_classroom_v7_1_continuous/generated（seed/split_seed 20260725）
layered: dataset_classroom_v7_1/generated（murmur seed 20260726）
train/valid/test: 8000/800/800（8.89/0.89/0.89 小时）
scene (train): hvac 3029, far 2018, machine 1838, identity 954, noise-only 161
murmur: 2455 条（PCAFETER 1233，PRESTO 1222），30% 非 identity/noise-only
near SNR low/main/high: hvac 626/1822/581, machine 354/1119/365（1.0-14.0，median ~6）
far SNR low/main/high:  477/1209/332（2.0-10.0，median 4.9）
speaker/room/noise/event split overlap: 0
19200 WAV audit（2000 抽样）: pass
```

验证域 `validation_domains_v7_1.json` 共七域，首次使用 `numeric_filters` 按
SNR 分档，把"必须降噪"和"不许伤人声"拆开考核：

```text
v7_1_near_denoise: hvac+machine、无 murmur、SNR<=8 dB，128 条，weight 1.25
v7_1_near_high:    hvac+machine、无 murmur、SNR>8 dB，64 条（do-no-harm）
v7_1_murmur:       有 murmur 层的非 identity 场景，128 条
v7_1_far:          far_speech 全部，128 条
zh_clean_raw / zh_clean_norm / voicebank: 与 v7 相同（含原门槛）
```

v5 epoch 3 在该验证域上的基线（`runs/classroom_v5_chinese/v7_1domains_baseline.json`）
与据此锚定的门槛（保护线风格同 v7：init 可过、selection 驱动提升）：

| domain | SI-SNR+ | PESQ+ | STOI+ | gate (si/pesq/stoi) |
|---|---:|---:|---:|---|
| near_denoise | +1.687 | +0.153 | +0.0260 | +1.20 / +0.10 / +0.005 |
| near_high | +0.966 | +0.260 | +0.0197 | -0.20 / +0.05 / -0.005 |
| murmur | +1.345 | +0.146 | +0.0195 | +0.80 / +0.08 / +0.005 |
| far | +0.378 | +0.093 | +0.0023 | 0.0 / +0.05 / -0.002 |
| zh_clean_raw | 94.95 dB | -0.015 | -0.0002 | 同 v7（60 dB / P10 30 dB） |
| zh_clean_norm | 82.04 dB | -0.062 | -0.0025 | 同 v7（55 dB / P10 20 dB） |
| voicebank | +4.881 | +0.412 | +0.0150 | 同 v7（+3.0 / +0.25 / 0.0） |

near_high 是唯一允许 SI-SNR 略负的域，专门盯住 17.23 左尾问题：训练变激进
后不能在安静样本上伤人声。

**不从头训练**：从 v5 epoch 3 加载权重、新建 optimizer（v7 血缘实验结论不变，
不使用 v6/v7 checkpoint）。lr 2e-5、20 epoch、patience 5、frozen BN、25%
VoiceBank replay、clean-fraction 0.12（identity）。epochs=0 加载检查已通过
（CUDA/STFT/全部数据路径/七域/replay/init）。

在 Windows CMD 的 `(work) D:\modeltraining\gtcrn>` 中粘贴下面完整一行：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v7_1\generated\train\noisy --train-clean ..\dataset_classroom_v7_1\generated\train\clean --valid-noisy ..\dataset_classroom_v7_1\generated\valid\noisy --valid-clean ..\dataset_classroom_v7_1\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v7_1\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v7_1\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.12 --clean-scene-type identity --epoch-size 12000 --validation-domains validation_domains_v7_1.json --out-dir runs\classroom_v7_1 --segment-seconds 4 --epochs 20 --batch-size 8 --lr 2e-5 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --freeze-batchnorm --save-every-epoch --early-stopping-patience 5 --init-checkpoint runs\classroom_v5_chinese\checkpoints\best.tar --seed 20260727 --overwrite-run
```

训练完成后下一步：best.tar 的 v7.1 test 对照（vs v5）、v4/v2/VoiceBank 回归、
AISHELL clean 透传、分档试听矩阵（重点 near_high 是否伤人声、near_denoise
降噪是否可闻）。

### 17.26 v7.1 正式训练结果与 test 评估（2026-07-20）

用户完成 v7.1 正式训练。训练在 epoch 17 因连续 5 个 epoch 没有改善而提前停止，
总耗时约 148.4 分钟。综合选择最优为 epoch 12：

```text
runs/classroom_v7_1/checkpoints/best.tar
checkpoint epoch: 12
selection loss: -3.01620983
```

epoch 12 后 near denoise、near high、murmur、far 仍有小幅变化，但综合 selection
loss 没有更好；正式评估使用 epoch 12 的 `best.tar`，不使用 `last.tar`。

在未用于选择 checkpoint 的 v7.1 test 800 条样本上，与同一 test 的 v5 epoch 3：

| metric | v5 | v7.1 best |
|---|---:|---:|
| SI-SNR change | +1.676 dB | +2.994 dB |
| PESQ change | +0.201 | +0.368 |
| STOI change | +0.0215 | +0.0577 |
| improved fraction | 85.23% | 94.88% |
| noise-only attenuation | 20.75 dB | 24.34 dB |
| clean enhanced SI-SNR | 80.33 dB | 84.88 dB |
| clean PESQ change | -0.0835 | -0.0630 |
| clean STOI change | -0.00269 | -0.00181 |

v7.1 在更符合用户听感的噪声强度上，比 v5 明显更积极；同时 clean PESQ/STOI
退化减轻，说明本轮没有简单用“更强降噪换人声损伤”。分场景 test：

| scene | files | SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|---:|
| HVAC | 297 | +3.624 | +0.420 | +0.0576 |
| machine/no-RIR | 198 | +2.934 | +0.398 | +0.0574 |
| far speech | 189 | +2.065 | +0.254 | +0.0580 |
| continuous without murmur | 480 | +2.905 | +0.376 | +0.0566 |
| student murmur | 204 | +3.201 | +0.348 | +0.0601 |

当前已完成：v7.1 test、v5 对照、训练验证域和基本分场景统计。`v4_test` 回归
进程仍在运行，当前 `metrics.csv` 为空，不能据此下结论；VoiceBank 验证域在训练
过程中保持通过（epoch 12 SI-SNR change +6.27 dB）。待回归完成后再决定是否提升
为部署候选。

### 17.27 runs 整理与当前试听候选（2026-07-20）

根目录散落的 v7.1 文件已整理为：

```text
runs/v7_1_eval/logs/
  v7_1_gen.log
  v7_1_murmur.log
  v7_1_baseline.log 与各评估日志
runs/v7_1_eval/provenance/v7_1_audit.json
```

当前版本判断：

```text
v5:       中文透明度较好，但 v7.1 噪声上的降噪较弱
v7:       使用了错误的较安静 SNR 正式集，不作为候选
v7.1:     当前最佳候选，正确覆盖较强教室底噪，best 为 epoch 12
v6:       包含用户否决的孤立事件声场，不作为候选
```

因此当前只需要听 v7.1 A/B：

```text
runs/v7_1_eval/listening/01_v5/
runs/v7_1_eval/listening/02_v7_1_best/
```

两边使用同一批 v7.1 test 文件和同名标签：01-03 HVAC low/main/high，04-06
机器 low/main/high，07-09 far low/main/high，10 学生讨论，11 identity。每组按
`_noisy.wav -> _enhanced.wav -> _clean.wav` 顺序听，重点比较 v5 与 v7.1 的
`_enhanced.wav`。v7.1 在指标上胜出，但是否最终采用仍以这些分档试听和 v4/v2
完整回归为准。

### 17.28 v7.1 分档试听结论与 v7.2 repair smoke（2026-07-20）

用户试听 v5/v7.1 同文件 A/B 后确认：03/06/09 high-SNR 和 11 identity 没有
人声损失；04 machine-low、07 far-low 有人声损失；01/04/07 low-SNR 的残余
噪声都不够干净。

三个失败样本虽然客观指标均改善，但不能推翻听感：

| sample | input SI-SNR | v7.1 SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|---:|
| 01 HVAC low | 4.08 | +8.11 | +0.200 | +0.0377 |
| 04 machine low | 3.98 | +2.20 | +0.084 | +0.0310 |
| 07 far low | 2.43 | +4.79 | +0.284 | +0.1022 |

因此 v7.1 是当前最佳候选，但仍未达到部署听感；不能继续仅靠增加训练轮数或
降低 SNR。v7.2 只验证两个假设：低 SNR far 不再同时强制去混响，以及对预测
语音幅度低于 clean 的频点增加非对称惩罚。

实现保持向后兼容：

```text
--far-preserve-rir-max-snr 6
  far 且 background SNR <= 6 dB 时，target_mode=rir_preserved；
  target 为无新增噪声但保留该 RIR 的 speech component。

--speech-underestimate-weight 1.0
  compressed magnitude 低于 clean 时使用 2 倍权重；默认 0 时与旧 HybridLoss
  数值完全一致。
```

v7.2 smoke 位于 `dataset_classroom_v7_2_smoke/generated`，400/80/80，沿用已
确认的 v7.1 噪声分布。审计通过；train 104 个 far 中 94 个为
`rir_preserved`、10 个 high-SNR far 保持 `native_room`；speaker/room/noise
跨 split overlap 为 0，1120 WAV 抽查全部通过。

验证配置 `validation_domains_v7_2_smoke.json` 分为 near low/main/high、
far-preserved、murmur、clean raw/norm 和 VoiceBank。v7.1 best 初始化基线已保存
到 `runs/v7_2_eval/provenance/v7_2_smoke_baseline.json`。其中 valid near-low
只有 2 条，因此本轮只能做方向性 smoke，不能把绝对数值当正式结论。

训练管线已用 `epochs=0` 检查通过，没有执行训练。用户在 Windows CMD 手动运行
3 epoch repair smoke：

```cmd
D:\Anaconda\Scripts\conda.exe run --no-capture-output -n work python train_custom.py --train-noisy ..\dataset_classroom_v7_2_smoke\generated\train\noisy --train-clean ..\dataset_classroom_v7_2_smoke\generated\train\clean --valid-noisy ..\dataset_classroom_v7_2_smoke\generated\valid\noisy --valid-clean ..\dataset_classroom_v7_2_smoke\generated\valid\clean --train-metadata-csv ..\dataset_classroom_v7_2_smoke\generated\metadata\train.csv --valid-metadata-csv ..\dataset_classroom_v7_2_smoke\generated\metadata\valid.csv --replay-train-noisy ..\dataset_voicebank_replay_v4\generated\train\noisy --replay-train-clean ..\dataset_voicebank_replay_v4\generated\train\clean --replay-train-manifest ..\dataset_voicebank_replay_v4\generated\metadata\train.json --replay-valid-noisy ..\dataset_voicebank_replay_v4\generated\valid\noisy --replay-valid-clean ..\dataset_voicebank_replay_v4\generated\valid\clean --replay-valid-manifest ..\dataset_voicebank_replay_v4\generated\metadata\valid.json --replay-fraction 0.25 --clean-fraction 0.15 --clean-scene-type identity --epoch-size 2000 --validation-domains validation_domains_v7_2_smoke.json --out-dir runs\classroom_v7_2_smoke --segment-seconds 4 --epochs 3 --batch-size 8 --lr 5e-6 --scheduler none --num-workers 4 --identity-loss-weight 0.1 --speech-underestimate-weight 1.0 --freeze-batchnorm --save-every-epoch --init-checkpoint runs\classroom_v7_1\checkpoints\best.tar --seed 20260730 --overwrite-run
```

训练后只比较 01/04/07 对应类型与 03/06/09/11 保护样本；若 04/07 人声仍损伤，
停止在当前 GTCRN/loss 上继续微调，进入更强模型或成熟持续噪声前端对照。

### 17.29 v7.2 repair smoke 训练结果与试听矩阵（2026-07-20）

注意：17.28 的 smoke 训练命令实际未被执行（`runs/classroom_v7_2_smoke` 只有
epochs=0 加载检查的 config.json，无 checkpoint）。本轮由助手补跑同一命令，
3 epoch 约 35 分钟，best.tar = epoch 2（selection -3.4354），全部八个域在
每个 epoch 均通过门槛。

与初始化（v7.1 best）在同一 smoke 验证域上的对照（base -> epoch 2）：

| domain | SI-SNR change | PESQ change | clean SI-SNR dB |
|---|---:|---:|---:|
| near_low (n=2) | +2.62 -> +2.66 | +0.125 -> +0.130 | — |
| near_main | +4.16 -> +4.09 | +0.403 -> +0.386 | — |
| near_high | +0.64 -> +0.71 | +0.374 -> +0.357 | — |
| far_preserved | +3.98 -> +3.93 | +0.223 -> +0.231 | — |
| murmur | +4.47 -> +4.41 | +0.280 -> +0.280 | — |
| voicebank | +5.44 -> +5.35 | +0.391 -> +0.383 | — |
| zh_clean_raw | — | -0.0119 -> -0.0053 | 97.27 -> 106.46 |
| zh_clean_norm | — | -0.0629 -> -0.0472 | 86.36 -> 92.96 |

3 epoch / lr 5e-6 只带来很小移动：noisy 域基本持平（±0.1 dB 内），但 clean
透明度明显上升（raw +9.2 dB、norm +6.6 dB，clean PESQ 损失约减半），与
speech-underestimate 惩罚的设计方向一致。far_preserved SI-SNR 微降是预期内
的口径变化（target 不再强制去混响），不能与 v7.1 直接比数值。

13 个试听文件（v7.1 test，同一批 noisy）两模型指标对照：

| tag | v7.1 SI-SNR+ | v7.2 SI-SNR+ | v7.1 PESQ+ | v7.2 PESQ+ |
|---|---:|---:|---:|---:|
| 01 hvac_low | +8.11 | +7.81 | +0.200 | +0.184 |
| 04 machine_low | +2.20 | +2.40 | +0.084 | +0.066 |
| 07 far_low | +4.79 | +4.72 | +0.284 | +0.291 |
| 11 identity | -86.45 | -48.96 | -0.009 | 0.000 |
| 12 near_high_worst | -7.24 | -6.66 | +0.097 | +0.117 |
| 13 overall_worst | -7.88 | -7.52 | +0.132 | +0.151 |

（02/03/05/06/08/09/10 均小幅变化，详见 `runs/v7_2_eval/listening13_metrics/`。）
identity 样本的 clean 抑制明显减轻（等效透传约 58 -> 96 dB），12/13 左尾也略
收窄；01 残余噪声指标略降（降噪稍保守）。03/06/09 保护样本保持正改善。

试听矩阵（A/B/C 同文件同名）：

```text
runs/v7_1_eval/listening/01_v5/
runs/v7_1_eval/listening/02_v7_1_best/
runs/v7_1_eval/listening/03_v7_2_smoke/
```

按 17.28 决策框架，只听 01/04/07（低 SNR 失败样本）与 03/06/09/11（保护样
本），重点回答两个问题：04/07 的人声损伤是否消失或减轻到可接受；01/04/07
的残余噪声是否比 02_v7_1_best 更干净或可接受。若 04/07 人声仍损伤，停止在
当前 GTCRN/loss 上继续微调，转入更强模型或成熟持续噪声前端对照；若通过，
再以 v7.2 两个开关（far-preserve-rir、speech-underestimate-weight）生成正式
数据并训练。

**复跑确认（2026-07-20）**：用户随后用同一命令自行重跑了一遍 3 epoch smoke
（`--overwrite-run` 覆盖助手那次）。同 seed 下指标逐域一致（差异 ≤0.01 dB，
为 GPU 非确定性噪声），唯一差别是 best.tar 由 epoch 2 变为 epoch 1
（selection 前三 epoch 相差 <0.08，属同一水平）。`03_v7_2_smoke` 试听矩阵与
13 文件指标已用当前 best.tar（epoch 1）重新导出，结论不变：01 +7.87、
04 +2.35、07 +4.74、11 identity -43.61（PESQ 0.000）、12 -6.75、13 -7.65。
这同时说明 v7.2 的 3 epoch smoke 在同 seed 下行为可复现。

### 17.30 v7.2 主观验收与音乐噪声分流（2026-07-23）

用户试听 v7.2 epoch 1 后确认：人声损伤已经消失；`04_machine_low` 仍有残余噪声，
并带有颗粒化、类似“音乐噪声”的质感。因此本轮只能得出两个分开的结论：

1. `rir_preserved + speech_underestimate_weight=1.0` 的保人声方向通过 smoke；
2. v7.2 没有解决低 SNR 持续机器噪声的自然度，不能据此直接扩大为正式训练。

这里的“音乐噪声”不是输入中有音乐，而是时频掩蔽随帧和频点波动，留下短促窄带
残留所产生的音调感。继续降低训练 SNR、提高抑制强度或盲目增加 epoch，通常会
扩大这种波动，并重新引入吞音风险，故暂不采用。

下一步先做无需训练的因果增益平滑小试验，只检查同一 `04` 的原始 v7.2 输出、
10/30 ms 轻平滑和 10/60 ms 较慢释放三个版本，同时用 `06_machine_high` 检查
平滑是否伤害原本正常的人声。试听文件位于
`runs/v7_2_eval/listening_gain_smoothing/`，每个目录按 `00_noisy ->
01_v7_2_original -> 02_mild_10_30ms -> 03_slow_release_10_60ms ->
04_energy_matched_10_30ms -> 05_clean`
顺序排列。该试验只决定后处理方向，不改变正式候选：

```text
正式模型候选：v7.1 epoch 12
保人声 repair 候选：v7.2 epoch 1（仅 smoke，不可单独发布）
暂停事项：扩大 v7.2 数据、从头训练、继续降低 SNR
```

另外，v7.1 的完整回归已经结束：v4 test 为 SI-SNR +0.252 dB / PESQ +0.218，
v2 test 为 SI-SNR +0.487 dB / PESQ +0.220，VoiceBank 为 SI-SNR +7.014 dB /
PESQ +0.538；三者均为正改善，且 v4/v2 clean PESQ 变化分别仅 -0.0028 和
-0.0221。它们说明 v7.1 没有明显破坏旧域，但不能推翻本轮对 `04` 音乐噪声的
主观否决。

**第一次平滑试听反馈（2026-07-23）**：用户确认 `04` 的 10/30 ms 与
10/60 ms 平滑均比原始 v7.2 自然，且 `06` 三个版本的人声都自然；但 `04`
平滑后相对 clean 有距离变远的感觉。频带能量检查确认这不只是低频问题：原始
v7.2 相对 clean 的 0-250 Hz 为 -1.98 dB、其余主要语音频带约 -0.24 至
-1.07 dB；10/30 ms 平滑后各频带变为约 -1.64 至 -2.84 dB，10/60 ms 则为
-3.52 至 -4.83 dB。也就是说，音乐噪声减少的同时发生了额外宽带衰减，慢释放
版本尤其明显。

因此不直接采用第一次平滑版本，也不据此修改训练集。下一小试验在 10/30 ms
平滑基础上逐帧匹配原始 v7.2 输出的频谱总能量，只保留频点间和时间上的平滑，
避免把整个人声继续压低。该能量匹配由 `smooth_enhancement_gain.py
--preserve-frame-energy` 实现，是因果逐帧运算，不需要查看未来帧。

能量匹配版本的检查结果符合设计：`04` 的整体电平由普通 10/30 ms 平滑的
-31.06 dBFS 恢复至 -29.79 dBFS，接近原始 v7.2 的 -29.72 dBFS；0-2 kHz
相对 clean 的分频段差值也由约 -1.64 至 -2.84 dB 恢复至 -0.39 至
-1.92 dB，接近原始输出的 -0.28 至 -1.98 dB。最终仍以试听为准：先比较
`01_v7_2_original.wav` 与 `04_energy_matched_10_30ms.wav` 的音乐噪声和距离感，
再在 `06` 目录确认 `04` 没有引入人声损伤。

**能量匹配试听结论（2026-07-23）**：用户确认能量匹配版与原始 v7.2 在 `04`
中听起来非常接近，两者降噪都不干净，并且人声都有“颤颤巍巍”的调制感；`06`
仍然正常。由此判定因果增益平滑分支不通过并停止，不再继续调 attack/release。
这说明 `04` 的问题已经存在于模型输出本身，而不是简单的输出响度或窄带残留：
低 SNR 持续机器噪声下，复数掩码对语音也产生了随时间变化的调制。

当前 `n_fft=256` GTCRN 只有 31,861 个参数；仓库中的官方 DNS3/VCTK checkpoint
使用独立的 `n_fft=512` 模型，共 48,245 个参数。下一步先用这两个现成官方模型
对同一 `04/06` 做零训练诊断，不能把它们直接视为部署候选，因为其 512/256
STFT（32 ms 窗、16 ms hop）与当前 160/80（10 ms 窗、5 ms hop）延迟不同：

```text
若官方模型没有颤音：优先怀疑当前 n_fft=256 容量/频率分辨率及自定义模型血缘；
若官方模型也有颤音：优先判定 GTCRN 复数掩码路线在该场景达到上限，转向更强模型；
无论哪种结果：不再扩大 v7.2，不从当前 v7.1/v7.2 checkpoint 继续微调。
```

`infer_custom.py` 已支持显式覆盖 STFT 参数，以便正确加载没有 `config` 字段的官方
checkpoint；旧自定义 checkpoint 不传覆盖参数时行为保持不变。

官方零训练对照已导出到 `runs/official_gtcrn_diagnostic/listening/`，每个目录按
`00_noisy -> 01_v7_2 -> 02_official_dns3 -> 03_official_vctk -> 04_clean` 排列。
同文件指标如下（这些指标不能替代颤音试听）：

| sample/model | SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|
| 04 / v7.2 | +6.141 dB | +0.794 | +0.0802 |
| 04 / official DNS3 | +6.473 dB | +0.643 | +0.0913 |
| 04 / official VCTK | +5.978 dB | +0.393 | +0.0665 |
| 06 / v7.2 | +4.017 dB | +0.289 | +0.0614 |
| 06 / official DNS3 | +5.259 dB | +0.631 | +0.0729 |
| 06 / official VCTK | -4.168 dB | -0.098 | -0.1490 |

DNS3 在两个文件的 SI-SNR/STOI 都高于 v7.2，具备试听价值；VCTK 在 `06` 明显
退化，不作为后续基座。若 DNS3 主观上也消除 `04` 颤音，下一步不是从随机权重
重训，而是设计独立的 DNS3 初始化、`n_fft=512` 对照链，并单独验证保持 5 ms
hop 的可行性；若 DNS3 仍颤，则停止 GTCRN 内部迭代并比较更强架构。

**官方 DNS3 主观结论（2026-07-24）**：用户确认 `02_official_dns3.wav` 比 v7.2
降噪更好，颗粒感小很多，声音更连贯、圆润；但仍有轻微不自然和忽强忽弱，类似
通话时对端信号不稳定。因此 DNS3 明显胜过当前自定义基座，但仍未通过最终听感
门槛，不能直接作为成品。

为避免把官方权重的改善误解为可以直接满足 5 ms hop，又对同一 `04/06` 做了
零训练 STFT 兼容性检查，目录为 `runs/dns3_stft_diagnostic/listening/`：

| sample/config | SI-SNR change | PESQ change | STOI change |
|---|---:|---:|---:|
| 04 / DNS3 native 32 ms / 16 ms | +6.473 dB | +0.643 | +0.0913 |
| 04 / DNS3 20 ms / 5 ms | +1.563 dB | +0.208 | -0.0135 |
| 04 / DNS3 10 ms / 5 ms | -7.024 dB | +0.018 | -0.1744 |
| 06 / DNS3 native 32 ms / 16 ms | +5.259 dB | +0.631 | +0.0729 |
| 06 / DNS3 20 ms / 5 ms | +1.158 dB | +0.219 | +0.0272 |
| 06 / DNS3 10 ms / 5 ms | -4.800 dB | -0.029 | -0.1191 |

结论：官方 DNS3 权重依赖其训练时的时间尺度，不能只修改 window/hop 直接部署；
低延迟配置若继续走 GTCRN 必须重新适配训练。由于原生 DNS3 仍有调制感，暂不
直接投入完整 v8 训练。下一步优先用同一 `04/06` 对比一个成熟的更强实时增强
模型；只有它在听感上明显胜出，才值得围绕该架构重建训练与部署链。若更强模型
也无法改善，则应回到目标 SNR 与“低噪声、绝对无损伤”的产品取舍，而不是继续
追求更强抑制。

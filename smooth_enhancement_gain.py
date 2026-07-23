import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_utils import stft_to_wav, wav_to_stft


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply causal attack/release smoothing to an enhanced/noisy STFT gain."
    )
    parser.add_argument("--noisy", required=True)
    parser.add_argument("--enhanced", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=160)
    parser.add_argument("--hop-length", type=int, default=80)
    parser.add_argument("--attack-ms", type=float, default=10.0)
    parser.add_argument("--release-ms", type=float, default=30.0)
    parser.add_argument("--max-gain", type=float, default=2.0)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def smoothing_coefficient(time_ms, hop_length, fs):
    if time_ms <= 0:
        return 0.0
    return math.exp(-hop_length / (fs * time_ms / 1000.0))


def main():
    args = parse_args()
    noisy, noisy_fs = sf.read(args.noisy, dtype="float32")
    enhanced, enhanced_fs = sf.read(args.enhanced, dtype="float32")
    if noisy_fs != args.fs or enhanced_fs != args.fs:
        raise ValueError(
            f"Expected {args.fs} Hz, got noisy={noisy_fs}, enhanced={enhanced_fs}"
        )
    if noisy.ndim != 1 or enhanced.ndim != 1:
        raise ValueError("Only mono WAV files are supported")

    length = min(len(noisy), len(enhanced))
    noisy_tensor = torch.from_numpy(noisy[:length])
    enhanced_tensor = torch.from_numpy(enhanced[:length])
    noisy_spec = wav_to_stft(
        noisy_tensor,
        args.n_fft,
        args.hop_length,
        args.win_length,
        center=args.center,
    )
    enhanced_spec = wav_to_stft(
        enhanced_tensor,
        args.n_fft,
        args.hop_length,
        args.win_length,
        center=args.center,
    )

    noisy_mag = torch.linalg.vector_norm(noisy_spec, dim=-1)
    enhanced_mag = torch.linalg.vector_norm(enhanced_spec, dim=-1)
    raw_gain = torch.clamp(enhanced_mag / (noisy_mag + 1e-7), 0.0, args.max_gain)

    attack = smoothing_coefficient(args.attack_ms, args.hop_length, args.fs)
    release = smoothing_coefficient(args.release_ms, args.hop_length, args.fs)
    smoothed_gain = torch.empty_like(raw_gain)
    smoothed_gain[:, 0] = raw_gain[:, 0]
    for frame in range(1, raw_gain.shape[1]):
        target = raw_gain[:, frame]
        previous = smoothed_gain[:, frame - 1]
        coefficient = torch.where(target < previous, attack, release)
        smoothed_gain[:, frame] = coefficient * previous + (1.0 - coefficient) * target

    scale = smoothed_gain / (raw_gain + 1e-7)
    smoothed_spec = enhanced_spec * scale[..., None]
    output = stft_to_wav(
        smoothed_spec,
        args.n_fft,
        args.hop_length,
        args.win_length,
        length=length,
        center=args.center,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.clip(output.numpy(), -1.0, 1.0), args.fs)


if __name__ == "__main__":
    main()

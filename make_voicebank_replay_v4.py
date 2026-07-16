import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["files"]


def read_mono(path, target_fs):
    wav, fs = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if fs != target_fs:
        divisor = math.gcd(fs, target_fs)
        wav = resample_poly(wav, target_fs // divisor, fs // divisor).astype(np.float32)
    return wav


def rms(wav):
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))


def speech_activity(wav, fs, threshold_dbfs):
    frame_samples = int(round(0.02 * fs))
    frame_count = len(wav) // frame_samples
    frames = wav[: frame_count * frame_samples].reshape(frame_count, frame_samples)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return float(np.mean(frame_rms >= 10.0 ** (threshold_dbfs / 20.0)))


def group_by_speaker(names):
    grouped = defaultdict(list)
    for name in names:
        grouped[Path(name).stem.split("_", 1)[0]].append(name)
    return {speaker: sorted(files) for speaker, files in grouped.items()}


def build_pair(grouped, noisy_root, clean_root, samples, args, rng):
    speaker = rng.choice(sorted(grouped))
    names = grouped[speaker].copy()
    rng.shuffle(names)
    noisy = np.zeros(samples, dtype=np.float32)
    clean = np.zeros(samples, dtype=np.float32)
    used = []
    cursor = 0
    for name in names[: args.max_stitched_files]:
        noisy_wav = read_mono(noisy_root / name, args.fs)
        clean_wav = read_mono(clean_root / name, args.fs)
        length = min(len(noisy_wav), len(clean_wav))
        noisy_wav, clean_wav = noisy_wav[:length], clean_wav[:length]
        remaining = samples - cursor
        if remaining <= 0:
            break
        if length > remaining:
            start = rng.randint(0, length - remaining)
            noisy_wav = noisy_wav[start : start + remaining]
            clean_wav = clean_wav[start : start + remaining]
        else:
            start = 0
        length = len(clean_wav)
        noisy[cursor : cursor + length] = noisy_wav
        clean[cursor : cursor + length] = clean_wav
        used.append((name, start, length))
        cursor += length
        if cursor < samples:
            cursor = min(
                samples,
                cursor + rng.randint(args.min_gap_samples, args.max_gap_samples),
            )
    clean -= float(np.mean(clean))
    noisy -= float(np.mean(noisy))
    target_dbfs = rng.uniform(args.clean_dbfs_min, args.clean_dbfs_max)
    gain = (10.0 ** (target_dbfs / 20.0)) / (rms(clean) + 1e-12)
    clean *= gain
    noisy *= gain
    peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))), 1e-6)
    if peak > args.peak_limit:
        gain = args.peak_limit / peak
        clean *= gain
        noisy *= gain
    return noisy.astype(np.float32), clean.astype(np.float32), speaker, used


def generate_split(split, count, names, args, rng):
    grouped = group_by_speaker(names)
    samples = int(round(args.segment_seconds * args.fs))
    split_root = Path(args.out_root) / split
    noisy_out, clean_out = split_root / "noisy", split_root / "clean"
    noisy_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)
    metadata_root = Path(args.out_root) / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    manifest, rows = [], []
    for index in range(count):
        best = None
        for _ in range(max(1, args.clean_file_attempts)):
            candidate = build_pair(
                grouped,
                Path(args.noisy_root),
                Path(args.clean_root),
                samples,
                args,
                rng,
            )
            activity = speech_activity(candidate[1], args.fs, args.speech_activity_dbfs)
            if best is None or activity > best[-1]:
                best = (*candidate, activity)
            if activity >= args.min_speech_activity:
                break
        noisy, clean, speaker, used, activity = best
        name = f"{split}_{index:06d}.wav"
        sf.write(noisy_out / name, noisy, args.fs, subtype="PCM_16")
        sf.write(clean_out / name, clean, args.fs, subtype="PCM_16")
        manifest.append(name)
        rows.append(
            {
                "file": name,
                "split": split,
                "speaker_id": speaker,
                "source_files": "|".join(item[0] for item in used),
                "source_spans": "|".join(f"{start}:{length}" for _, start, length in used),
                "speech_activity": activity,
                "segment_seconds": args.segment_seconds,
                "fs": args.fs,
            }
        )
        if (index + 1) % args.log_interval == 0 or index + 1 == count:
            print(f"{split}: generated {index + 1}/{count}")
    with open(metadata_root / f"{split}.json", "w", encoding="utf-8") as handle:
        json.dump({"files": manifest}, handle, indent=2)
    with open(metadata_root / f"{split}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create 4-second VoiceBank replay pairs.")
    parser.add_argument("--noisy-root", default=r"..\dataset\train\noisy")
    parser.add_argument("--clean-root", default=r"..\dataset\train\clean")
    parser.add_argument("--train-manifest", default=r"..\dataset\splits\voicebank_serious\train.json")
    parser.add_argument("--valid-manifest", default=r"..\dataset\splits\voicebank_serious\valid.json")
    parser.add_argument("--out-root", default=r"..\dataset_voicebank_replay_v4\generated")
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-valid", type=int, default=1000)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--max-stitched-files", type=int, default=4)
    parser.add_argument("--clean-file-attempts", type=int, default=10)
    parser.add_argument("--speech-activity-dbfs", type=float, default=-40.0)
    parser.add_argument("--min-speech-activity", type=float, default=0.4)
    parser.add_argument("--gap-min-seconds", type=float, default=0.08)
    parser.add_argument("--gap-max-seconds", type=float, default=0.30)
    parser.add_argument("--clean-dbfs-min", type=float, default=-28.0)
    parser.add_argument("--clean-dbfs-max", type=float, default=-18.0)
    parser.add_argument("--peak-limit", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.min_gap_samples = int(round(args.gap_min_seconds * args.fs))
    args.max_gap_samples = int(round(args.gap_max_seconds * args.fs))

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.rglob("*.wav")) and not args.overwrite:
        raise FileExistsError(f"{out_root} already contains wav files")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    train_names = read_manifest(args.train_manifest)
    valid_names = read_manifest(args.valid_manifest)
    rng = random.Random(args.seed)
    generate_split("train", args.num_train, train_names, args, rng)
    generate_split("valid", args.num_valid, valid_names, args, rng)
    config = vars(args).copy()
    config["train_source_files"] = len(train_names)
    config["valid_source_files"] = len(valid_names)
    with open(out_root / "metadata" / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()

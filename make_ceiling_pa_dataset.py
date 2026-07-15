import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def list_wavs(root):
    return sorted(Path(root).rglob("*.wav"))


def read_mono(path, target_fs):
    wav, fs = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if fs != target_fs:
        divisor = math.gcd(fs, target_fs)
        wav = resample_poly(wav, target_fs // divisor, fs // divisor).astype(np.float32)
    return wav


def rms(wav):
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)) + 1e-12))


def rms_dbfs(wav):
    return 20.0 * math.log10(rms(wav) + 1e-12)


def scale_to_dbfs(wav, target_dbfs):
    gain = (10.0 ** (target_dbfs / 20.0)) / (rms(wav) + 1e-12)
    return (wav * gain).astype(np.float32)


def take_segment(wav, samples, rng):
    if len(wav) >= samples:
        start = rng.randint(0, len(wav) - samples)
        return wav[start : start + samples], start
    repeats = int(math.ceil(samples / max(1, len(wav))))
    tiled = np.tile(wav, repeats)
    return tiled[:samples], 0


def take_segment_from_range(wav, samples, rng, start_fraction=0.0, end_fraction=1.0):
    if len(wav) < samples:
        return take_segment(wav, samples, rng)
    max_start = len(wav) - samples
    range_start = min(max_start, max(0, int(round(len(wav) * start_fraction))))
    range_end = min(max_start, max(range_start, int(round(len(wav) * end_fraction)) - samples))
    start = rng.randint(range_start, range_end) if range_end > range_start else range_start
    return wav[start : start + samples], start


def prepare_rir(rir):
    if len(rir) == 0:
        return None, 0
    peak = int(np.argmax(np.abs(rir)))
    rir = rir / (np.sqrt(np.sum(rir.astype(np.float64) ** 2)) + 1e-8)
    return rir.astype(np.float32), peak


def convolve_aligned(clean, rir, direct_index):
    reverberant = fftconvolve(clean, rir, mode="full").astype(np.float32)
    end = direct_index + len(clean)
    if len(reverberant) < end:
        reverberant = np.pad(reverberant, (0, end - len(reverberant)))
    return reverberant[direct_index:end].astype(np.float32)


def estimate_rir_metrics(rir, direct_index, fs):
    energy = np.square(rir.astype(np.float64))
    total_energy = float(np.sum(energy)) + 1e-12
    direct_radius = max(1, int(round(0.0025 * fs)))
    direct_start = max(0, direct_index - direct_radius)
    direct_end = min(len(rir), direct_index + direct_radius + 1)
    direct_energy = float(np.sum(energy[direct_start:direct_end])) + 1e-12
    reverb_energy = max(1e-12, total_energy - direct_energy)
    drr_db = 10.0 * math.log10(direct_energy / reverb_energy)

    schroeder = np.cumsum(energy[::-1])[::-1]
    decay_db = 10.0 * np.log10(np.maximum(schroeder, 1e-20) / max(schroeder[0], 1e-20))
    times = np.arange(len(rir), dtype=np.float64) / fs
    mask = (decay_db <= -5.0) & (decay_db >= -35.0)
    rt60 = float("nan")
    if np.count_nonzero(mask) >= 20:
        slope, _ = np.polyfit(times[mask], decay_db[mask], 1)
        if slope < -1e-6:
            rt60 = float(-60.0 / slope)
    return rt60, drr_db


def apply_rir(clean, rir_path, fs, early_reflections_ms):
    rir = read_mono(rir_path, fs)
    rir, direct_index = prepare_rir(rir)
    if rir is None:
        return clean.copy(), clean.copy(), 0, float("nan"), float("nan")
    reverberant = convolve_aligned(clean, rir, direct_index)
    early_rir = rir.copy()
    early_end = min(
        len(early_rir),
        direct_index + max(1, int(round(early_reflections_ms * fs / 1000.0))),
    )
    early_rir[early_end:] = 0.0
    early_target = convolve_aligned(clean, early_rir, direct_index)
    rt60, drr_db = estimate_rir_metrics(rir, direct_index, fs)
    return reverberant, early_target, direct_index, rt60, drr_db


def mix_at_snr(speech, noise, snr_db):
    speech_rms = rms(speech)
    noise_rms = rms(noise)
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    return (noise * (target_noise_rms / (noise_rms + 1e-12))).astype(np.float32)


def sample_snr_db(args, rng):
    if args.snr_profile == "quiet_classroom":
        roll = rng.random()
        if roll < 0.75:
            return rng.uniform(20.0, 30.0)
        if roll < 0.95:
            return rng.uniform(15.0, 20.0)
        return rng.uniform(10.0, 15.0)
    if args.snr_profile == "classroom":
        roll = rng.random()
        if roll < 0.75:
            return rng.uniform(15.0, 30.0)
        if roll < 0.95:
            return rng.uniform(10.0, 15.0)
        return rng.uniform(6.0, 10.0)
    return rng.uniform(args.snr_min, args.snr_max)


def speech_activity_ratio(wav, fs, threshold_dbfs=-40.0):
    frame_samples = max(1, int(round(0.02 * fs)))
    if len(wav) < frame_samples:
        return float(rms_dbfs(wav) >= threshold_dbfs)
    frame_count = len(wav) // frame_samples
    frames = wav[: frame_count * frame_samples].reshape(frame_count, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    frame_dbfs = 20.0 * np.log10(frame_rms + 1e-12)
    return float(np.mean(frame_dbfs >= threshold_dbfs))


def take_speech_segment(
    wav,
    samples,
    rng,
    target_dbfs,
    fs,
    threshold_dbfs,
    min_activity,
    attempts,
):
    best = None
    for _ in range(max(1, attempts)):
        candidate, start = take_segment(wav, samples, rng)
        candidate = candidate - float(np.mean(candidate))
        candidate = scale_to_dbfs(candidate, target_dbfs)
        activity = speech_activity_ratio(candidate, fs, threshold_dbfs)
        if best is None or activity > best[2]:
            best = (candidate, start, activity)
        if activity >= min_activity:
            return candidate, start, activity
    return best


def sample_scene_type(profile, rng):
    if profile == "legacy":
        return "legacy"
    weighted = [
        ("clean", 0.10),
        ("reverb_only", 0.15),
        ("reverb_noise", 0.60),
        ("noise_no_reverb", 0.10),
        ("noise_only", 0.05),
    ]
    roll = rng.random()
    cumulative = 0.0
    for name, weight in weighted:
        cumulative += weight
        if roll < cumulative:
            return name
    return weighted[-1][0]


def choose_split_files(clean_root):
    clean_root = Path(clean_root)
    return {
        "train": list_wavs(clean_root / "train"),
        "valid": list_wavs(clean_root / "dev"),
        "test": list_wavs(clean_root / "test"),
    }


def speaker_id(path):
    return path.stem.split("_", 1)[0]


def choose_voicebank_split_files(clean_root, test_clean_root, valid_speaker_fraction, seed):
    clean_files = list_wavs(clean_root)
    if not clean_files:
        raise ValueError(f"No VoiceBank train wav files found under {clean_root}.")

    speakers = sorted({speaker_id(path) for path in clean_files})
    shuffled = speakers.copy()
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, int(math.ceil(len(speakers) * valid_speaker_fraction)))
    valid_speakers = set(shuffled[:valid_count])

    train_files = [path for path in clean_files if speaker_id(path) not in valid_speakers]
    valid_files = [path for path in clean_files if speaker_id(path) in valid_speakers]
    test_files = list_wavs(test_clean_root) if test_clean_root else []
    if not test_files:
        test_files = valid_files
    return {
        "train": train_files,
        "valid": valid_files,
        "test": test_files,
    }


def choose_clean_files(args):
    clean_root = Path(args.clean_root)
    if args.clean_layout == "aishell":
        return choose_split_files(clean_root)
    if args.clean_layout == "voicebank":
        return choose_voicebank_split_files(
            clean_root,
            Path(args.test_clean_root) if args.test_clean_root else None,
            args.valid_speaker_fraction,
            args.split_seed,
        )

    aishell_like = {
        "train": list_wavs(clean_root / "train"),
        "valid": list_wavs(clean_root / "dev"),
        "test": list_wavs(clean_root / "test"),
    }
    if aishell_like["train"] and aishell_like["valid"] and aishell_like["test"]:
        return aishell_like

    prepared_like = {
        "train": list_wavs(clean_root / "train" / "clean"),
        "valid": list_wavs(clean_root / "valid" / "clean"),
        "test": list_wavs(clean_root / "valid" / "clean"),
    }
    if prepared_like["train"] and prepared_like["valid"]:
        return prepared_like

    return choose_voicebank_split_files(
        clean_root,
        Path(args.test_clean_root) if args.test_clean_root else None,
        args.valid_speaker_fraction,
        args.split_seed,
    )


def gather_noise_files(musan_root, rirs_root, profile, noise_roots):
    if profile == "presto_pcafeter":
        noise_files = []
        for root in noise_roots:
            root = Path(root)
            if root.name.upper() in {"PRESTO", "PCAFETER"}:
                noise_files.extend(list_wavs(root))
        return sorted(noise_files)

    if profile == "classroom":
        weights = {
            "OOFFICE": 14,
            "PRESTO": 2,
            "OMEETING": 1,
            "PCAFETER": 1,
        }
        noise_files = []
        for root in noise_roots:
            root = Path(root)
            root_files = list_wavs(root)
            repeat = weights.get(root.name.upper(), 1)
            noise_files.extend(root_files * repeat)
        return sorted(noise_files)

    if profile == "custom_indoor":
        noise_files = []
        for root in noise_roots:
            noise_files.extend(list_wavs(root))
        return sorted(noise_files)

    noise_files = []
    if profile == "broad":
        for subdir in ["noise", "music", "speech"]:
            noise_files.extend(list_wavs(Path(musan_root) / subdir))
        noise_files.extend(list_wavs(Path(rirs_root) / "pointsource_noises"))
        return sorted(noise_files)

    if profile == "rir_noise_only":
        musan_subdirs = []
    else:
        musan_subdirs = ["noise"]

    for subdir in musan_subdirs:
        noise_files.extend(list_wavs(Path(musan_root) / subdir))

    real_noise_root = Path(rirs_root) / "real_rirs_isotropic_noises"
    noise_files.extend(
        path for path in list_wavs(real_noise_root) if "noise" in path.name.lower()
    )
    return sorted(noise_files)


def gather_rir_files(rirs_root, room_types, layout):
    rirs_root = Path(rirs_root)
    if layout == "but" or (layout == "auto" and not (rirs_root / "simulated_rirs").is_dir()):
        return sorted(
            path
            for path in list_wavs(rirs_root)
            if "RIR" in path.parts and path.name.lower().startswith("ir_sweep")
        )
    rir_files = []
    simulated_root = rirs_root / "simulated_rirs"
    for room_type in room_types:
        rir_files.extend(list_wavs(simulated_root / room_type))
    return sorted(rir_files)


def split_counts(item_count, valid_fraction, test_fraction):
    if item_count < 3:
        raise ValueError("At least three independent groups are required for train/valid/test isolation.")
    valid_count = max(1, int(round(item_count * valid_fraction)))
    test_count = max(1, int(round(item_count * test_fraction)))
    if valid_count + test_count >= item_count:
        valid_count = 1
        test_count = 1
    return item_count - valid_count - test_count, valid_count, test_count


def split_rirs_by_room(rir_files, rirs_root, seed, valid_fraction, test_fraction):
    rirs_root = Path(rirs_root).resolve()
    grouped = defaultdict(list)
    for path in rir_files:
        relative = Path(path).resolve().relative_to(rirs_root)
        grouped[relative.parts[0]].append(path)
    rooms = sorted(grouped)
    random.Random(seed).shuffle(rooms)
    train_count, valid_count, _ = split_counts(len(rooms), valid_fraction, test_fraction)
    room_splits = {
        "train": rooms[:train_count],
        "valid": rooms[train_count : train_count + valid_count],
        "test": rooms[train_count + valid_count :],
    }
    return {
        split: sorted(path for room in split_rooms for path in grouped[room])
        for split, split_rooms in room_splits.items()
    }, room_splits


def split_noise_files_by_scene(noise_files, seed, valid_fraction, test_fraction):
    grouped = defaultdict(list)
    for path in noise_files:
        grouped[Path(path).parent.name.upper()].append(path)
    split_files = {"train": [], "valid": [], "test": []}
    split_names = {"train": [], "valid": [], "test": []}
    for scene_index, (scene, files) in enumerate(sorted(grouped.items())):
        files = sorted(files)
        random.Random(seed + scene_index * 1009).shuffle(files)
        train_count, valid_count, _ = split_counts(len(files), valid_fraction, test_fraction)
        selected = {
            "train": files[:train_count],
            "valid": files[train_count : train_count + valid_count],
            "test": files[train_count + valid_count :],
        }
        for split, paths in selected.items():
            split_files[split].extend(paths)
            split_names[split].extend(f"{scene}/{Path(path).name}" for path in paths)
    return {split: sorted(paths) for split, paths in split_files.items()}, split_names


def source_room_id(path, rirs_root):
    if not path:
        return ""
    return Path(path).resolve().relative_to(Path(rirs_root).resolve()).parts[0]


def scene_name(path, dataset_root):
    relative = Path(os.path.relpath(path, dataset_root))
    return relative.parts[0] if relative.parts else ""


def save_manifest(path, files):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"files": files}, handle, indent=2)


def generate_split(split, count, clean_files, noise_files, rir_files, args, rng):
    segment_samples = int(round(args.segment_seconds * args.fs))
    split_root = Path(args.out_root) / split
    clean_out = split_root / "clean"
    noisy_out = split_root / "noisy"
    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)

    metadata_dir = Path(args.out_root) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    metadata_rows = []

    for index in range(count):
        scene_type = sample_scene_type(args.scene_profile, rng)
        has_speech = scene_type != "noise_only"
        use_rir = scene_type in {"reverb_only", "reverb_noise"}
        use_noise = scene_type in {"reverb_noise", "noise_no_reverb", "noise_only"}
        if scene_type == "legacy":
            has_speech = True
            use_rir = rng.random() < args.rir_prob
            use_noise = True

        clean_path = None
        clean_start = 0
        if has_speech:
            best_clean = None
            for _ in range(max(1, args.clean_file_attempts)):
                candidate_path = rng.choice(clean_files)
                clean_raw = read_mono(candidate_path, args.fs)
                candidate, candidate_start, candidate_activity = take_speech_segment(
                    clean_raw,
                    segment_samples,
                    rng,
                    rng.uniform(args.clean_dbfs_min, args.clean_dbfs_max),
                    args.fs,
                    args.speech_activity_dbfs,
                    args.min_speech_activity,
                    args.segment_attempts,
                )
                if best_clean is None or candidate_activity > best_clean[3]:
                    best_clean = (
                        candidate_path,
                        candidate,
                        candidate_start,
                        candidate_activity,
                    )
                if candidate_activity >= args.min_speech_activity:
                    break
            clean_path, clean, clean_start, activity = best_clean
        else:
            clean = np.zeros(segment_samples, dtype=np.float32)
            activity = 0.0

        rir_path = rng.choice(rir_files) if use_rir else None
        rt60_estimate = float("nan")
        drr_db = float("nan")
        if rir_path is None:
            reverberant = clean.copy()
            target = clean.copy()
            rir_direct_index = 0
        else:
            reverberant, early_target, rir_direct_index, rt60_estimate, drr_db = apply_rir(
                clean, rir_path, args.fs, args.early_reflections_ms
            )
            target = early_target if args.target_mode == "early_reflections" else clean.copy()

        noise_path = rng.choice(noise_files) if use_noise else None
        noise_start = 0
        snr_db = float("nan")
        if noise_path is None:
            noisy = reverberant.astype(np.float32)
        else:
            noise_raw = read_mono(noise_path, args.fs)
            noise_time_ranges = {
                "train": (0.0, 0.70),
                "valid": (0.70, 0.85),
                "test": (0.85, 1.0),
            }
            noise_start_fraction, noise_end_fraction = noise_time_ranges[split]
            noise, noise_start = take_segment_from_range(
                noise_raw,
                segment_samples,
                rng,
                noise_start_fraction,
                noise_end_fraction,
            )
            noise = noise - float(np.mean(noise))
            if has_speech:
                snr_db = sample_snr_db(args, rng)
                scaled_noise = mix_at_snr(reverberant, noise, snr_db)
            else:
                scaled_noise = scale_to_dbfs(
                    noise, rng.uniform(args.noise_only_dbfs_min, args.noise_only_dbfs_max)
                )
            noisy = (reverberant + scaled_noise).astype(np.float32)

        peak = max(float(np.max(np.abs(target))), float(np.max(np.abs(noisy))), 1e-6)
        if peak > args.peak_limit:
            gain = args.peak_limit / peak
            target = (target * gain).astype(np.float32)
            noisy = (noisy * gain).astype(np.float32)

        name = f"{split}_{index:06d}.wav"
        sf.write(clean_out / name, target, args.fs, subtype="PCM_16")
        sf.write(noisy_out / name, noisy, args.fs, subtype="PCM_16")
        manifest_files.append(name)
        metadata_rows.append(
            {
                "file": name,
                "split": split,
                "scene_type": scene_type,
                "target_mode": args.target_mode,
                "clean_file": "" if clean_path is None else os.path.relpath(clean_path, args.dataset_root),
                "clean_start_sample": clean_start,
                "noise_file": "" if noise_path is None else os.path.relpath(noise_path, args.dataset_root),
                "noise_class": "" if noise_path is None else noise_path.parent.name.upper(),
                "noise_start_sample": noise_start,
                "noise_time_partition": (
                    "" if noise_path is None else f"{noise_start_fraction:.2f}-{noise_end_fraction:.2f}"
                ),
                "rir_file": "" if rir_path is None else os.path.relpath(rir_path, args.dataset_root),
                "room_id": source_room_id(rir_path, args.rirs_root),
                "rir_direct_index": rir_direct_index,
                "rt60_estimate_s": "" if math.isnan(rt60_estimate) else rt60_estimate,
                "drr_db": "" if math.isnan(drr_db) else drr_db,
                "snr_db": "" if math.isnan(snr_db) else snr_db,
                "speech_activity": activity,
                "clean_dbfs": rms_dbfs(target),
                "noisy_dbfs": rms_dbfs(noisy),
                "segment_seconds": args.segment_seconds,
                "fs": args.fs,
            }
        )

        if (index + 1) % args.log_interval == 0 or index + 1 == count:
            print(f"{split}: generated {index + 1}/{count}")

    save_manifest(metadata_dir / f"{split}.json", manifest_files)
    with open(metadata_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired clean/noisy data for ceiling-mic local-PA GTCRN training."
    )
    parser.add_argument("--dataset-root", default=r"..\dataset")
    parser.add_argument("--clean-root", default=r"..\dataset\clean_trainset_28spk_wav")
    parser.add_argument("--test-clean-root", default=r"..\dataset\clean_testset_wav")
    parser.add_argument(
        "--clean-layout",
        choices=["auto", "aishell", "voicebank"],
        default="auto",
        help="auto detects AISHELL-style train/dev/test, prepared train/valid/clean, or VoiceBank flat files.",
    )
    parser.add_argument("--valid-speaker-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--musan-root", default=r"..\dataset\musan")
    parser.add_argument("--rirs-root", default=r"..\dataset\BUT_ReverbDB_rel_19_06_RIR-Only")
    parser.add_argument("--rir-layout", choices=["auto", "but", "rirs_noise"], default="auto")
    parser.add_argument(
        "--noise-roots",
        nargs="+",
        default=[
            r"..\dataset\PRESTO",
            r"..\dataset\PCAFETER",
        ],
    )
    parser.add_argument(
        "--rir-room-types",
        nargs="+",
        default=["smallroom", "mediumroom"],
        help="Room folders under RIRS_NOISES/simulated_rirs to sample.",
    )
    parser.add_argument("--out-root", default=r"..\dataset_classroom_v2\generated")
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-valid", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument(
        "--snr-profile",
        choices=["quiet_classroom", "classroom", "uniform"],
        default="quiet_classroom",
        help=(
            "quiet_classroom: 75% 20-30 dB, 20% 15-20 dB, 5% 10-15 dB; "
            "classroom: 75% 15-30 dB, 20% 10-15 dB, 5% 6-10 dB; "
            "uniform: sample uniformly from --snr-min to --snr-max."
        ),
    )
    parser.add_argument("--clean-dbfs-min", type=float, default=-28.0)
    parser.add_argument("--clean-dbfs-max", type=float, default=-18.0)
    parser.add_argument("--rir-prob", type=float, default=0.9)
    parser.add_argument("--rir-valid-room-fraction", type=float, default=0.2)
    parser.add_argument("--rir-test-room-fraction", type=float, default=0.2)
    parser.add_argument("--noise-valid-file-fraction", type=float, default=0.2)
    parser.add_argument("--noise-test-file-fraction", type=float, default=0.2)
    parser.add_argument("--scene-profile", choices=["classroom_v2", "legacy"], default="classroom_v2")
    parser.add_argument("--target-mode", choices=["dry", "early_reflections"], default="early_reflections")
    parser.add_argument("--early-reflections-ms", type=float, default=50.0)
    parser.add_argument("--speech-activity-dbfs", type=float, default=-40.0)
    parser.add_argument("--min-speech-activity", type=float, default=0.4)
    parser.add_argument("--segment-attempts", type=int, default=10)
    parser.add_argument("--clean-file-attempts", type=int, default=10)
    parser.add_argument("--noise-only-dbfs-min", type=float, default=-38.0)
    parser.add_argument("--noise-only-dbfs-max", type=float, default=-28.0)
    parser.add_argument(
        "--noise-profile",
        choices=["presto_pcafeter", "classroom", "custom_indoor", "indoor", "broad", "rir_noise_only"],
        default="presto_pcafeter",
        help=(
            "presto_pcafeter: only PRESTO and PCAFETER under --noise-roots; "
            "classroom: mostly OOFFICE, with small PRESTO/OMEETING/PCAFETER fractions; "
            "custom_indoor: wav files under --noise-roots; "
            "indoor: MUSAN/noise plus RIRS files with noise in the name; "
            "broad: also include MUSAN music/speech and RIRS pointsource noises; "
            "rir_noise_only: only RIRS files with noise in the name."
        ),
    )
    parser.add_argument("--peak-limit", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.rglob("*.wav")) and not args.overwrite:
        raise FileExistsError(f"{out_root} already contains wav files. Use --overwrite or another --out-root.")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    clean_by_split = choose_clean_files(args)
    noise_files = gather_noise_files(
        args.musan_root,
        args.rirs_root,
        args.noise_profile,
        [Path(root) for root in args.noise_roots],
    )
    rir_files = gather_rir_files(args.rirs_root, args.rir_room_types, args.rir_layout)
    if not clean_by_split["train"] or not clean_by_split["valid"] or not clean_by_split["test"]:
        raise ValueError("Clean train/valid/test wav files were not found. Check --clean-root and --clean-layout.")
    if not noise_files:
        raise ValueError("No noise wav files found.")
    if not rir_files:
        raise ValueError("No RIR wav files found. Check --rirs-root and --rir-room-types.")

    print(f"clean train files: {len(clean_by_split['train'])}")
    print(f"clean valid files: {len(clean_by_split['valid'])}")
    print(f"clean test files: {len(clean_by_split['test'])}")
    noise_by_split, noise_names_by_split = split_noise_files_by_scene(
        noise_files,
        args.split_seed,
        args.noise_valid_file_fraction,
        args.noise_test_file_fraction,
    )
    rir_by_split, room_splits = split_rirs_by_room(
        rir_files,
        args.rirs_root,
        args.split_seed,
        args.rir_valid_room_fraction,
        args.rir_test_room_fraction,
    )
    print(f"unique noise files: {len(set(noise_files))}")
    print(f"rir files: {len(rir_files)}")
    for split in ["train", "valid", "test"]:
        print(
            f"{split} sources: rooms={len(room_splits[split])}, "
            f"rirs={len(rir_by_split[split])}, noises={len(noise_by_split[split])}"
        )
    print(f"out root: {out_root}")

    rng = random.Random(args.seed)
    generate_split(
        "train", args.num_train, clean_by_split["train"], noise_by_split["train"], rir_by_split["train"], args, rng
    )
    generate_split(
        "valid", args.num_valid, clean_by_split["valid"], noise_by_split["valid"], rir_by_split["valid"], args, rng
    )
    generate_split(
        "test", args.num_test, clean_by_split["test"], noise_by_split["test"], rir_by_split["test"], args, rng
    )

    config = vars(args).copy()
    config["clean_train_files"] = len(clean_by_split["train"])
    config["clean_valid_files"] = len(clean_by_split["valid"])
    config["clean_test_files"] = len(clean_by_split["test"])
    config["unique_noise_files"] = len(set(noise_files))
    config["rir_files"] = len(rir_files)
    config["rir_rooms_by_split"] = room_splits
    config["noise_files_by_split"] = noise_names_by_split
    with open(out_root / "metadata" / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()

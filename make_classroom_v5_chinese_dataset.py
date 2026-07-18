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
from scipy.signal import fftconvolve

from make_ceiling_pa_dataset import (
    estimate_rir_metrics,
    list_wavs,
    rms,
    rms_dbfs,
    save_manifest,
    scale_to_dbfs,
    speech_activity_ratio,
    split_counts,
    take_segment,
)
from make_classroom_v4_dataset import (
    NoiseEntry,
    RirEntry,
    gather_but_rirs,
    gather_rirs_real,
    gather_rirs_simulated,
    load_esc_entries,
    ms_category,
    prepare_background,
    prepare_event,
    prepare_rir,
    read_audio,
    relative_path,
    scale_event,
    split_entries_by_group,
    split_files_by_category,
    split_rirs_by_source,
    weighted_source_choice,
)


# 17.4 scene distribution:
#   10% native_room identity (half keep the low native level)
#   25% air-conditioner / office / fan-like additive noise
#   30% mild far-distance + short classroom RIR + quiet background
#   15% desk/chair/footstep/keyboard/door events
#   15% additive noise without extra RIR
#    5% noise-only
SCENE_WEIGHTS = [
    ("identity", 0.10),
    ("hvac_noise", 0.25),
    ("far_speech", 0.30),
    ("event", 0.15),
    ("noise_no_rir", 0.15),
    ("noise_only", 0.05),
]

RIR_SOURCE_WEIGHTS = {"but": 0.40, "rirs_real": 0.20, "rirs_sim": 0.40}
HVAC_SOURCE_WEIGHTS = {"ms_ac": 0.50, "ooffice": 0.50}
BACKGROUND_SOURCE_WEIGHTS = {"ms_continuous": 0.45, "ooffice": 0.35, "esc_background": 0.20}
EVENT_SOURCE_WEIGHTS = {"ms_event": 0.50, "esc_event": 0.50}

MS_AC_CATEGORIES = {"AirConditioner"}
MS_CONTINUOUS_CATEGORIES = {
    "AirConditioner",
    "CopyMachine",
    "Hallway",
    "Kitchen",
    "LivingRoom",
    "Office",
    "VacuumCleaner",
    "WasherDryer",
    "Washing",
}
MS_EVENT_CATEGORIES = {"SqueakyChair", "Typing"}
ESC_BACKGROUND_CATEGORIES = {
    "clock_tick",
    "rain",
    "vacuum_cleaner",
    "washing_machine",
    "wind",
}
ESC_EVENT_CATEGORIES = {"footsteps", "door_wood_knock"}


def load_transcript_ids(transcript_path):
    ids = set()
    with open(transcript_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split(None, 1)
            if fields:
                ids.add(fields[0])
    return ids


def gather_aishell_splits(aishell_root, transcript_ids):
    """Use the official speaker-isolated train/dev/test splits as-is."""
    root = Path(aishell_root)
    split_dirs = {"train": "train", "valid": "dev", "test": "test"}
    speaker_groups = {}
    excluded = []
    for split, folder in split_dirs.items():
        grouped = defaultdict(list)
        for path in list_wavs(root / folder):
            if path.stem not in transcript_ids:
                excluded.append(
                    {"split": split, "speaker": path.parent.name, "file": path.name,
                     "reason": "no_transcript"}
                )
                continue
            grouped[path.parent.name].append(path)
        speaker_groups[split] = {
            speaker: sorted(paths) for speaker, paths in sorted(grouped.items())
        }
    return speaker_groups, excluded


def build_native_speech(speaker_groups, samples, args, rng):
    """Stitch same-speaker utterances at their native level (no dBFS rescale)."""
    best = None
    speakers = sorted(speaker_groups)
    for _ in range(max(1, args.clean_file_attempts)):
        speaker = rng.choice(speakers)
        files = speaker_groups[speaker]
        order = files.copy()
        rng.shuffle(order)
        segment = np.zeros(samples, dtype=np.float32)
        used = []
        cursor = 0
        file_index = 0
        while cursor < samples and file_index < min(len(order), args.max_stitched_files):
            path = order[file_index]
            file_index += 1
            wav, _ = read_audio(path, args.fs)
            wav = wav - float(np.mean(wav))
            if not len(wav):
                continue
            remaining = samples - cursor
            if len(wav) > remaining:
                start = rng.randint(0, len(wav) - remaining)
                wav = wav[start : start + remaining]
            else:
                start = 0
            segment[cursor : cursor + len(wav)] = wav
            used.append((path, start, len(wav)))
            cursor += len(wav)
            if cursor < samples:
                gap = rng.randint(args.min_gap_samples, args.max_gap_samples)
                cursor = min(samples, cursor + gap)
        segment = segment - float(np.mean(segment))
        native_dbfs = rms_dbfs(segment)
        # Speech activity is measured at a fixed reference level so that quiet
        # native utterances are not rejected only for being quiet.
        reference = scale_to_dbfs(segment, args.reference_dbfs)
        activity = speech_activity_ratio(
            reference, args.fs, threshold_dbfs=args.speech_activity_dbfs
        )
        candidate = (segment, used, activity, speaker, native_dbfs)
        if best is None or activity > best[2]:
            best = candidate
        if activity >= args.min_speech_activity:
            break
    return best


def sample_scene_type(rng, scene_weights):
    roll = rng.random()
    cumulative = 0.0
    for name, weight in scene_weights:
        cumulative += weight
        if roll < cumulative:
            return name
    return scene_weights[-1][0]


def sample_snr_db(args, rng, offset_db=0.0):
    if rng.random() < args.snr_main_fraction:
        return rng.uniform(args.snr_main_min, args.snr_main_max) + offset_db
    return rng.uniform(args.snr_low_min, args.snr_low_max) + offset_db


def sample_scene_snr_db(scene_type, args, rng):
    if scene_type == "far_speech" and args.far_snr_main_min is not None:
        if rng.random() < args.far_snr_main_fraction:
            return rng.uniform(args.far_snr_main_min, args.far_snr_main_max)
        return rng.uniform(args.far_snr_low_min, args.far_snr_low_max)
    offsets = {
        "far_speech": args.far_snr_offset_db,
        "hvac_noise": args.hvac_snr_offset_db,
        "noise_no_rir": args.background_snr_offset_db,
    }
    return sample_snr_db(args, rng, offsets.get(scene_type, 0.0))


def speech_level_dbfs(scene_type, args, rng):
    if scene_type == "identity":
        if rng.random() < args.identity_native_fraction:
            return None, "native"
        return rng.uniform(args.identity_dbfs_min, args.identity_dbfs_max), "normalized"
    if scene_type == "far_speech":
        return rng.uniform(args.far_dbfs_min, args.far_dbfs_max), "far"
    return rng.uniform(args.speech_dbfs_min, args.speech_dbfs_max), "standard"


def apply_short_rir(dry, entry, args, rng):
    rir, direct_index, channel = prepare_rir(entry, args, rng)
    reverberant = fftconvolve(dry, rir, mode="full").astype(np.float32)
    end = direct_index + len(dry)
    if len(reverberant) < end:
        reverberant = np.pad(reverberant, (0, end - len(reverberant)))
    reverberant = reverberant[direct_index:end]
    rt60, drr = estimate_rir_metrics(rir, direct_index, args.fs)
    return reverberant, rt60, drr, direct_index, channel


def sample_short_rir(dry, rir_pools, args, rng):
    for _ in range(max(1, args.rir_attempts)):
        source = weighted_source_choice(rir_pools, RIR_SOURCE_WEIGHTS, rng)
        entry = rng.choice(rir_pools[source])
        reverberant, rt60, drr, direct_index, channel = apply_short_rir(
            dry, entry, args, rng
        )
        if not math.isnan(rt60) and args.rt60_min <= rt60 <= args.rt60_max:
            return source, entry, (reverberant, rt60, drr, direct_index, channel)
    raise RuntimeError(
        f"Could not sample an RIR with RT60 in [{args.rt60_min}, {args.rt60_max}] "
        f"after {args.rir_attempts} attempts"
    )


def mix_wet_dry(dry, wet, wet_mix):
    mixed = (1.0 - wet_mix) * dry + wet_mix * wet
    # Keep the speech level defined by the dry component: added reverb must not
    # by itself make the utterance louder.
    return (mixed * (rms(dry) / (rms(mixed) + 1e-12))).astype(np.float32)


def gather_ms_entries(root, categories, seed):
    root = Path(root)
    train = [
        path
        for path in list_wavs(root / "noise_train")
        if ms_category(path) in categories
    ]
    test = [
        path
        for path in list_wavs(root / "noise_test")
        if ms_category(path) in categories
    ]
    valid, final_test = split_files_by_category(test, seed)
    return {
        split: [
            NoiseEntry(path, "ms_snsd", ms_category(path), f"ms:{path.name}")
            for path in paths
        ]
        for split, paths in {"train": train, "valid": valid, "test": final_test}.items()
    }


def gather_ms_pools(root, seed):
    """Split MS-SNSD once over the category union so a file can never land in
    different splits through overlapping semantic pools."""
    union = MS_AC_CATEGORIES | MS_CONTINUOUS_CATEGORIES | MS_EVENT_CATEGORIES
    entries = gather_ms_entries(root, union, seed)
    pools = {}
    for split, split_entries in entries.items():
        pools[split] = {
            "ms_ac": [e for e in split_entries if e.category in MS_AC_CATEGORIES],
            "ms_continuous": [
                e for e in split_entries if e.category in MS_CONTINUOUS_CATEGORIES
            ],
            "ms_event": [e for e in split_entries if e.category in MS_EVENT_CATEGORIES],
        }
    return pools


def gather_ooffice_entries(root, seed, valid_fraction, test_fraction):
    entries = [
        NoiseEntry(path, "ooffice", "Office", f"ooffice:{path.name}")
        for path in list_wavs(root)
    ]
    return split_entries_by_group(entries, seed, valid_fraction, test_fraction)


def gather_noise_pools(args):
    hvac = {split: defaultdict(list) for split in ["train", "valid", "test"]}
    background = {split: defaultdict(list) for split in ["train", "valid", "test"]}
    events = {split: defaultdict(list) for split in ["train", "valid", "test"]}

    ms_pools = gather_ms_pools(args.ms_snsd_root, args.split_seed)
    ooffice = gather_ooffice_entries(
        args.ooffice_root,
        args.split_seed + 3000,
        args.noise_valid_file_fraction,
        args.noise_test_file_fraction,
    )
    esc_background = load_esc_entries(
        args.esc50_root, ESC_BACKGROUND_CATEGORIES, "esc_background"
    )
    esc_events = load_esc_entries(args.esc50_root, ESC_EVENT_CATEGORIES, "esc_event")

    for split in ["train", "valid", "test"]:
        hvac[split]["ms_ac"] = ms_pools[split]["ms_ac"]
        hvac[split]["ooffice"] = ooffice[split]
        background[split]["ms_continuous"] = ms_pools[split]["ms_continuous"]
        background[split]["ooffice"] = ooffice[split]
        background[split]["esc_background"] = esc_background[split]
        events[split]["ms_event"] = ms_pools[split]["ms_event"]
        events[split]["esc_event"] = esc_events[split]
    return hvac, background, events


def pick_noise(pools, weights, rng):
    source = weighted_source_choice(pools, weights, rng)
    return source, rng.choice(pools[source])


def generate_split(split, count, speaker_groups, rir_pools, hvac, background, events, args, rng):
    samples = int(round(args.segment_seconds * args.fs))
    split_root = Path(args.out_root) / split
    clean_out = split_root / "clean"
    noisy_out = split_root / "noisy"
    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path(args.out_root) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest, rows = [], []

    for index in range(count):
        scene_type = sample_scene_type(rng, args.scene_weights)
        has_speech = scene_type != "noise_only"

        if has_speech:
            native, used_clean, activity, speaker, native_dbfs = build_native_speech(
                speaker_groups, samples, args, rng
            )
            level, level_mode = speech_level_dbfs(scene_type, args, rng)
            if level is None:
                target = native.copy()
            else:
                target = scale_to_dbfs(native, level)
        else:
            target = np.zeros(samples, dtype=np.float32)
            used_clean, activity, speaker, native_dbfs, level_mode = [], 0.0, "", float("nan"), ""

        rir_entry = None
        rir_source = ""
        rir_channel = 0
        direct_index = 0
        rt60, drr = float("nan"), float("nan")
        wet_mix = ""
        if scene_type == "far_speech":
            rir_source, rir_entry, rir_result = sample_short_rir(native, rir_pools, args, rng)
            reverberant, rt60, drr, direct_index, rir_channel = rir_result
            wet_mix_value = rng.uniform(args.wet_mix_min, args.wet_mix_max)
            speech_component = mix_wet_dry(native, reverberant, wet_mix_value)
            # Far speech: the speech component and the target drop in level
            # together; the model is not asked to restore near-talk loudness.
            speech_component = speech_component * (
                rms(target) / (rms(speech_component) + 1e-12)
            )
            wet_mix = wet_mix_value
            noisy = speech_component.astype(np.float32)
            noise_source, noise_entry = pick_noise(background, BACKGROUND_SOURCE_WEIGHTS, rng)
            background_wav, noise_start = prepare_background(noise_entry, samples, args, rng)
            background_snr = sample_scene_snr_db(scene_type, args, rng)
            target_noise_rms = rms(noisy) / (10.0 ** (background_snr / 20.0))
            noisy = noisy + background_wav * (target_noise_rms / (rms(background_wav) + 1e-12))
        elif scene_type == "identity":
            noisy = target.copy()
            noise_source, noise_entry, noise_start = "", None, 0
            background_snr = float("nan")
        else:
            noisy = target.copy()
            noise_source, noise_entry, noise_start = "", None, 0
            background_snr = float("nan")
            if scene_type in {"hvac_noise", "noise_no_rir"}:
                pools = hvac if scene_type == "hvac_noise" else background
                weights = HVAC_SOURCE_WEIGHTS if scene_type == "hvac_noise" else BACKGROUND_SOURCE_WEIGHTS
                noise_source, noise_entry = pick_noise(pools, weights, rng)
                background_wav, noise_start = prepare_background(noise_entry, samples, args, rng)
                background_snr = sample_scene_snr_db(scene_type, args, rng)
                target_noise_rms = rms(noisy) / (10.0 ** (background_snr / 20.0))
                noisy = noisy + background_wav * (target_noise_rms / (rms(background_wav) + 1e-12))
            elif scene_type == "noise_only":
                noise_source, noise_entry = pick_noise(background, BACKGROUND_SOURCE_WEIGHTS, rng)
                background_wav, noise_start = prepare_background(noise_entry, samples, args, rng)
                noisy = scale_to_dbfs(
                    background_wav,
                    rng.uniform(args.noise_only_dbfs_min, args.noise_only_dbfs_max),
                )

        event_entry = None
        event_source = ""
        event_offset = 0
        event_length = 0
        event_snr = float("nan")
        if scene_type == "event":
            event_source, event_entry = pick_noise(events, EVENT_SOURCE_WEIGHTS, rng)
            event, event_offset, event_length = prepare_event(event_entry, samples, args, rng)
            event_snr = rng.uniform(args.event_snr_min, args.event_snr_max)
            noisy = noisy + scale_event(target, event, event_length, event_snr, args)

        peak = max(float(np.max(np.abs(target))), float(np.max(np.abs(noisy))), 1e-6)
        final_gain = 1.0
        if peak > args.peak_limit:
            final_gain = args.peak_limit / peak
            target = (target * final_gain).astype(np.float32)
            noisy = (noisy * final_gain).astype(np.float32)

        name = f"{split}_{index:06d}.wav"
        sf.write(clean_out / name, target, args.fs, subtype="PCM_16")
        sf.write(noisy_out / name, noisy, args.fs, subtype="PCM_16")
        manifest.append(name)
        rows.append(
            {
                "file": name,
                "split": split,
                "scene_type": scene_type,
                "target_mode": "native_room",
                "clean_files": "|".join(relative_path(path, args.dataset_root) for path, _, _ in used_clean),
                "clean_spans": "|".join(f"{start}:{length}" for _, start, length in used_clean),
                "speaker_id": speaker,
                "native_dbfs": "" if math.isnan(native_dbfs) else native_dbfs,
                "level_mode": level_mode,
                "speech_activity": activity,
                "noise_file": "" if noise_entry is None else relative_path(noise_entry.path, args.dataset_root),
                "noise_source": noise_source,
                "noise_category": "" if noise_entry is None else noise_entry.category,
                "noise_start_sample": noise_start,
                "background_snr_db": "" if math.isnan(background_snr) else background_snr,
                "event_file": "" if event_entry is None else relative_path(event_entry.path, args.dataset_root),
                "event_source": event_source,
                "event_category": "" if event_entry is None else event_entry.category,
                "event_offset_sample": event_offset,
                "event_length_sample": event_length,
                "event_snr_db": "" if math.isnan(event_snr) else event_snr,
                "rir_file": "" if rir_entry is None else relative_path(rir_entry.path, args.dataset_root),
                "rir_source": rir_source,
                "room_id": "" if rir_entry is None else rir_entry.room_id,
                "rir_channel": rir_channel,
                "rir_direct_index": direct_index,
                "rt60_estimate_s": "" if math.isnan(rt60) else rt60,
                "drr_db": "" if math.isnan(drr) else drr,
                "wet_mix": wet_mix,
                "clean_dbfs": rms_dbfs(target),
                "noisy_dbfs": rms_dbfs(noisy),
                "final_gain": final_gain,
                "segment_seconds": args.segment_seconds,
                "fs": args.fs,
            }
        )
        if (index + 1) % args.log_interval == 0 or index + 1 == count:
            print(f"{split}: generated {index + 1}/{count}")

    save_manifest(metadata_dir / f"{split}.json", manifest)
    with open(metadata_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate classroom_v5_chinese paired data from AISHELL-1 native_room speech."
    )
    parser.add_argument("--dataset-root", default=r"..\dataset")
    parser.add_argument("--aishell-root", default=r"..\dataset\data_aishell\wav_extracted")
    parser.add_argument(
        "--transcript",
        default=r"..\dataset\data_aishell\transcript\aishell_transcript_v0.8.txt",
    )
    parser.add_argument("--but-rirs-root", default=r"..\dataset\BUT_ReverbDB_rel_19_06_RIR-Only")
    parser.add_argument("--rirs-noises-root", default=r"..\dataset\RIRS_NOISES")
    parser.add_argument("--ms-snsd-root", default=r"..\dataset\MS-SNSD-sparse")
    parser.add_argument("--ooffice-root", default=r"..\dataset\OOFFICE")
    parser.add_argument("--esc50-root", default=r"..\dataset\ESC-50-master")
    parser.add_argument("--out-root", default=r"..\dataset_classroom_v5_chinese\generated")
    parser.add_argument("--num-train", type=int, default=8000)
    parser.add_argument("--num-valid", type=int, default=800)
    parser.add_argument("--num-test", type=int, default=800)
    parser.add_argument("--identity-fraction", type=float, default=0.10)
    parser.add_argument("--hvac-fraction", type=float, default=0.25)
    parser.add_argument("--far-speech-fraction", type=float, default=0.30)
    parser.add_argument("--event-fraction", type=float, default=0.15)
    parser.add_argument("--noise-no-rir-fraction", type=float, default=0.15)
    parser.add_argument("--noise-only-fraction", type=float, default=0.05)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--reference-dbfs", type=float, default=-25.0)
    parser.add_argument("--speech-activity-dbfs", type=float, default=-40.0)
    parser.add_argument("--min-speech-activity", type=float, default=0.4)
    parser.add_argument("--clean-file-attempts", type=int, default=10)
    parser.add_argument("--max-stitched-files", type=int, default=3)
    parser.add_argument("--gap-min-seconds", type=float, default=0.08)
    parser.add_argument("--gap-max-seconds", type=float, default=0.30)
    parser.add_argument("--identity-native-fraction", type=float, default=0.5)
    parser.add_argument("--identity-dbfs-min", type=float, default=-28.0)
    parser.add_argument("--identity-dbfs-max", type=float, default=-22.0)
    parser.add_argument("--speech-dbfs-min", type=float, default=-33.0)
    parser.add_argument("--speech-dbfs-max", type=float, default=-24.0)
    parser.add_argument("--far-dbfs-min", type=float, default=-40.0)
    parser.add_argument("--far-dbfs-max", type=float, default=-32.0)
    parser.add_argument("--rt60-min", type=float, default=0.15)
    parser.add_argument("--rt60-max", type=float, default=0.55)
    parser.add_argument("--wet-mix-min", type=float, default=0.25)
    parser.add_argument("--wet-mix-max", type=float, default=0.60)
    parser.add_argument("--rir-attempts", type=int, default=60)
    parser.add_argument("--snr-main-fraction", type=float, default=0.75)
    parser.add_argument("--snr-main-min", type=float, default=12.0)
    parser.add_argument("--snr-main-max", type=float, default=22.0)
    parser.add_argument("--snr-low-min", type=float, default=8.0)
    parser.add_argument("--snr-low-max", type=float, default=12.0)
    parser.add_argument("--far-snr-offset-db", type=float, default=0.0)
    parser.add_argument("--hvac-snr-offset-db", type=float, default=0.0)
    parser.add_argument("--background-snr-offset-db", type=float, default=0.0)
    parser.add_argument("--far-snr-main-fraction", type=float, default=0.75)
    parser.add_argument("--far-snr-main-min", type=float, default=None)
    parser.add_argument("--far-snr-main-max", type=float, default=None)
    parser.add_argument("--far-snr-low-min", type=float, default=None)
    parser.add_argument("--far-snr-low-max", type=float, default=None)
    parser.add_argument("--noise-only-dbfs-min", type=float, default=-38.0)
    parser.add_argument("--noise-only-dbfs-max", type=float, default=-28.0)
    parser.add_argument("--event-snr-min", type=float, default=12.0)
    parser.add_argument("--event-snr-max", type=float, default=24.0)
    parser.add_argument("--event-peak-ratio", type=float, default=0.8)
    parser.add_argument("--event-trim-threshold", type=float, default=0.02)
    parser.add_argument("--event-margin-seconds", type=float, default=0.05)
    parser.add_argument("--but-area-min", type=float, default=25.0)
    parser.add_argument("--but-area-max", type=float, default=100.0)
    parser.add_argument("--but-height-max", type=float, default=4.5)
    parser.add_argument("--sim-length-min", type=float, default=5.0)
    parser.add_argument("--sim-length-max", type=float, default=15.0)
    parser.add_argument("--sim-width-min", type=float, default=4.0)
    parser.add_argument("--sim-width-max", type=float, default=10.0)
    parser.add_argument("--sim-height-min", type=float, default=2.5)
    parser.add_argument("--sim-height-max", type=float, default=4.2)
    parser.add_argument("--sim-area-min", type=float, default=40.0)
    parser.add_argument("--sim-area-max", type=float, default=100.0)
    parser.add_argument("--rir-valid-room-fraction", type=float, default=0.2)
    parser.add_argument("--rir-test-room-fraction", type=float, default=0.2)
    parser.add_argument("--noise-valid-file-fraction", type=float, default=0.2)
    parser.add_argument("--noise-test-file-fraction", type=float, default=0.2)
    parser.add_argument("--peak-limit", type=float, default=0.98)
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.scene_weights = [
        ("identity", args.identity_fraction),
        ("hvac_noise", args.hvac_fraction),
        ("far_speech", args.far_speech_fraction),
        ("event", args.event_fraction),
        ("noise_no_rir", args.noise_no_rir_fraction),
        ("noise_only", args.noise_only_fraction),
    ]
    if any(weight < 0.0 for _, weight in args.scene_weights):
        raise ValueError("Scene fractions must be non-negative")
    scene_weight_sum = sum(weight for _, weight in args.scene_weights)
    if not math.isclose(scene_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Scene fractions must sum to 1.0, got {scene_weight_sum}")
    far_snr_values = [
        args.far_snr_main_min,
        args.far_snr_main_max,
        args.far_snr_low_min,
        args.far_snr_low_max,
    ]
    if any(value is not None for value in far_snr_values):
        if any(value is None for value in far_snr_values):
            raise ValueError("All four dedicated far SNR bounds must be provided together")
        if not 0.0 <= args.far_snr_main_fraction <= 1.0:
            raise ValueError("--far-snr-main-fraction must be in [0, 1]")
        if not (
            args.far_snr_low_min <= args.far_snr_low_max
            <= args.far_snr_main_min <= args.far_snr_main_max
        ):
            raise ValueError("Far SNR bounds must satisfy low_min <= low_max <= main_min <= main_max")
    args.min_gap_samples = int(round(args.gap_min_seconds * args.fs))
    args.max_gap_samples = int(round(args.gap_max_seconds * args.fs))

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.rglob("*.wav")) and not args.overwrite:
        raise FileExistsError(f"{out_root} already contains wav files")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    transcript_ids = load_transcript_ids(args.transcript)
    speaker_groups, excluded = gather_aishell_splits(args.aishell_root, transcript_ids)
    for split in ["train", "valid", "test"]:
        speakers = speaker_groups[split]
        print(
            f"{split}: speakers={len(speakers)}, "
            f"utterances={sum(len(paths) for paths in speakers.values())}"
        )
    print(f"excluded without transcript: {len(excluded)}")

    rir_entries = []
    rir_entries.extend(gather_but_rirs(args.but_rirs_root, args))
    rir_entries.extend(gather_rirs_real(args.rirs_noises_root))
    rir_entries.extend(gather_rirs_simulated(args.rirs_noises_root, args))
    rir_pools, room_splits = split_rirs_by_source(
        rir_entries,
        args.split_seed,
        args.rir_valid_room_fraction,
        args.rir_test_room_fraction,
    )
    hvac, background, events = gather_noise_pools(args)

    for split in ["train", "valid", "test"]:
        print(
            f"{split}: rirs={sum(len(v) for v in rir_pools[split].values())}, "
            f"hvac={sum(len(v) for v in hvac[split].values())}, "
            f"background={sum(len(v) for v in background[split].values())}, "
            f"events={sum(len(v) for v in events[split].values())}"
        )

    rng = random.Random(args.seed)
    for split, count in [
        ("train", args.num_train),
        ("valid", args.num_valid),
        ("test", args.num_test),
    ]:
        generate_split(
            split,
            count,
            speaker_groups[split],
            rir_pools[split],
            hvac[split],
            background[split],
            events[split],
            args,
            rng,
        )

    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "excluded_no_transcript.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "speaker", "file", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    config = vars(args).copy()
    config["scene_weights"] = args.scene_weights
    config["rir_source_weights"] = RIR_SOURCE_WEIGHTS
    config["hvac_source_weights"] = HVAC_SOURCE_WEIGHTS
    config["background_source_weights"] = BACKGROUND_SOURCE_WEIGHTS
    config["event_source_weights"] = EVENT_SOURCE_WEIGHTS
    config["ms_ac_categories"] = sorted(MS_AC_CATEGORIES)
    config["ms_continuous_categories"] = sorted(MS_CONTINUOUS_CATEGORIES)
    config["ms_event_categories"] = sorted(MS_EVENT_CATEGORIES)
    config["esc_background_categories"] = sorted(ESC_BACKGROUND_CATEGORIES)
    config["esc_event_categories"] = sorted(ESC_EVENT_CATEGORIES)
    config["transcript_ids"] = len(transcript_ids)
    config["excluded_no_transcript"] = len(excluded)
    config["speakers_by_split"] = {
        split: sorted(speaker_groups[split]) for split in ["train", "valid", "test"]
    }
    config["rooms_by_split"] = {
        split: {source: rooms for source, rooms in sources.items()}
        for split, sources in room_splits.items()
    }
    config["rir_counts_by_split"] = {
        split: {source: len(entries) for source, entries in sources.items()}
        for split, sources in rir_pools.items()
    }
    config["hvac_counts_by_split"] = {
        split: {source: len(entries) for source, entries in sources.items()}
        for split, sources in hvac.items()
    }
    config["background_counts_by_split"] = {
        split: {source: len(entries) for source, entries in sources.items()}
        for split, sources in background.items()
    }
    config["event_counts_by_split"] = {
        split: {source: len(entries) for source, entries in sources.items()}
        for split, sources in events.items()
    }
    with open(metadata_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()

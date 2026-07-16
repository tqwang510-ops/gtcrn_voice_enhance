import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from make_ceiling_pa_dataset import (
    choose_clean_files,
    estimate_rir_metrics,
    list_wavs,
    rms,
    rms_dbfs,
    sample_scene_type,
    sample_snr_db,
    save_manifest,
    scale_to_dbfs,
    speech_activity_ratio,
    split_counts,
    take_segment,
)


RIR_SOURCE_WEIGHTS = {"but": 0.40, "rirs_real": 0.20, "rirs_sim": 0.40}
NOISE_SOURCE_WEIGHTS = {
    "ms_snsd": 0.55,
    "presto_pcafeter": 0.25,
    "rirs_isotropic": 0.10,
    "esc_background": 0.10,
}
MS_SNSD_CATEGORIES = {
    "AirConditioner",
    "Babble",
    "Cafe",
    "CafeTeria",
    "CopyMachine",
    "Hallway",
    "Kitchen",
    "LivingRoom",
    "Neighbor",
    "NeighborSpeaking",
    "Office",
    "Restaurant",
    "Typing",
    "VacuumCleaner",
    "WasherDryer",
    "Washing",
}
ESC_BACKGROUND_CATEGORIES = {
    "clock_tick",
    "rain",
    "vacuum_cleaner",
    "washing_machine",
    "wind",
}
ESC_EVENT_CATEGORIES = {
    "can_opening",
    "clapping",
    "clock_alarm",
    "coughing",
    "door_wood_creaks",
    "door_wood_knock",
    "footsteps",
    "keyboard_typing",
    "mouse_click",
    "pouring_water",
    "sneezing",
    "water_drops",
}


@dataclass(frozen=True)
class RirEntry:
    path: Path
    source: str
    room_id: str


@dataclass(frozen=True)
class NoiseEntry:
    path: Path
    source: str
    category: str
    group_id: str


def read_audio(path, target_fs, rng=None, select_channel=False):
    wav, fs = sf.read(path, dtype="float32", always_2d=True)
    channel = 0
    if select_channel and wav.shape[1] > 1:
        channel = rng.randrange(wav.shape[1])
        wav = wav[:, channel]
    else:
        wav = wav.mean(axis=1)
    if fs != target_fs:
        divisor = math.gcd(fs, target_fs)
        wav = resample_poly(wav, target_fs // divisor, fs // divisor).astype(np.float32)
    return wav.astype(np.float32), channel


def weighted_source_choice(pools, weights, rng):
    available = [(name, weight) for name, weight in weights.items() if pools.get(name)]
    if not available:
        raise ValueError("No source pools are available for weighted sampling")
    threshold = rng.random() * sum(weight for _, weight in available)
    cumulative = 0.0
    for name, weight in available:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return available[-1][0]


def speaker_id(path):
    return Path(path).stem.split("_", 1)[0]


def group_clean_by_speaker(clean_files):
    grouped = defaultdict(list)
    for path in clean_files:
        grouped[speaker_id(path)].append(Path(path))
    return {speaker: sorted(paths) for speaker, paths in grouped.items()}


def build_stitched_speech(speaker_groups, samples, args, rng):
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
        segment = scale_to_dbfs(
            segment - float(np.mean(segment)),
            rng.uniform(args.clean_dbfs_min, args.clean_dbfs_max),
        )
        activity = speech_activity_ratio(
            segment, args.fs, threshold_dbfs=args.speech_activity_dbfs
        )
        candidate = (segment, used, activity, speaker)
        if best is None or activity > best[2]:
            best = candidate
        if activity >= args.min_speech_activity:
            break
    return best


def parse_list_entries(list_path, marker, root_parent, source):
    entries = []
    pattern = re.compile(rf"--{marker}\s+(\S+).*?\s(RIRS_NOISES[/\\].+\.wav)$")
    for line in Path(list_path).read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        room_id, relative = match.groups()
        path = Path(root_parent) / Path(relative.replace("/", os.sep))
        entries.append((path, room_id, source))
    return entries


def read_but_room_dimensions(room_root):
    values = {}
    for line in (Path(room_root) / "env_meta.txt").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        fields = line.split("\t", 1)
        if len(fields) == 2:
            values[fields[0].lstrip("$")] = fields[1]
    try:
        return (
            float(values["EnvDepth"]),
            float(values["EnvWidth"]),
            float(values["EnvHeight"]),
        )
    except (KeyError, ValueError):
        return None


def gather_but_rirs(root, args):
    root = Path(root)
    allowed_rooms = set()
    for room_root in root.iterdir():
        if not room_root.is_dir():
            continue
        dimensions = read_but_room_dimensions(room_root)
        if dimensions is None:
            continue
        length, width, height = dimensions
        area = length * width
        if (
            args.but_area_min <= area <= args.but_area_max
            and height <= args.but_height_max
        ):
            allowed_rooms.add(room_root.name)
    return [
        RirEntry(path, "but", f"but:{path.relative_to(root).parts[0]}")
        for path in list_wavs(root)
        if "RIR" in path.parts
        and path.name.lower().startswith("ir_sweep")
        and path.relative_to(root).parts[0] in allowed_rooms
    ]


def normalize_rirs_real_room(room_id):
    lower = room_id.lower()
    if lower.startswith("air_binaural_meeting"):
        return "air_meeting"
    if lower.startswith("air_binaural_office"):
        return "air_office"
    if lower in {
        "rvb2014_smallroom1",
        "rvb2014_smallroom2",
        "rvb2014_mediumroom1",
        "rvb2014_mediumroom2",
    }:
        return lower
    if lower.startswith("rwcp_cirline_ofc"):
        return "rwcp_office"
    return None


def gather_rirs_real(root):
    root = Path(root)
    list_path = root / "real_rirs_isotropic_noises" / "rir_list"
    parsed = parse_list_entries(list_path, "room-id", root.parent, "rirs_real")
    entries = []
    for path, room_id, source in parsed:
        normalized_room = normalize_rirs_real_room(room_id)
        if path.exists() and normalized_room:
            entries.append(RirEntry(path, source, f"rirs_real:{normalized_room}"))
    return entries


def gather_rirs_simulated(root, args):
    root = Path(root) / "simulated_rirs"
    entries = []
    for room_type in ["smallroom", "mediumroom"]:
        room_root = root / room_type
        for line in (room_root / "room_info").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 4:
                continue
            name, length, width, height = fields[:4]
            length, width, height = float(length), float(width), float(height)
            area = length * width
            if not (
                args.sim_length_min <= length <= args.sim_length_max
                and args.sim_width_min <= width <= args.sim_width_max
                and args.sim_height_min <= height <= args.sim_height_max
                and args.sim_area_min <= area <= args.sim_area_max
            ):
                continue
            folder_name = name.split("-", 1)[-1]
            room_id = f"rirs_sim:{room_type}:{folder_name}"
            entries.extend(
                RirEntry(path, "rirs_sim", room_id)
                for path in list_wavs(room_root / folder_name)
            )
    return entries


def split_rirs_by_source(entries, seed, valid_fraction, test_fraction):
    by_source_room = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        by_source_room[entry.source][entry.room_id].append(entry)
    pools = {split: defaultdict(list) for split in ["train", "valid", "test"]}
    room_splits = {split: defaultdict(list) for split in pools}
    for source_index, (source, rooms_to_entries) in enumerate(sorted(by_source_room.items())):
        rooms = sorted(rooms_to_entries)
        random.Random(seed + source_index * 1009).shuffle(rooms)
        train_count, valid_count, _ = split_counts(
            len(rooms), valid_fraction, test_fraction
        )
        selected = {
            "train": rooms[:train_count],
            "valid": rooms[train_count : train_count + valid_count],
            "test": rooms[train_count + valid_count :],
        }
        for split, split_rooms in selected.items():
            room_splits[split][source] = split_rooms
            for room in split_rooms:
                pools[split][source].extend(rooms_to_entries[room])
    return pools, room_splits


def ms_category(path):
    return re.sub(r"_\d+$", "", Path(path).stem)


def split_entries_by_group(entries, seed, valid_fraction, test_fraction):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry.group_id].append(entry)
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    train_count, valid_count, _ = split_counts(len(groups), valid_fraction, test_fraction)
    selections = {
        "train": groups[:train_count],
        "valid": groups[train_count : train_count + valid_count],
        "test": groups[train_count + valid_count :],
    }
    return {
        split: [entry for group in split_groups for entry in grouped[group]]
        for split, split_groups in selections.items()
    }


def split_files_by_category(files, seed):
    grouped = defaultdict(list)
    for path in files:
        grouped[ms_category(path)].append(Path(path))
    valid, test = [], []
    for index, (_, paths) in enumerate(sorted(grouped.items())):
        paths = sorted(paths)
        random.Random(seed + index * 1013).shuffle(paths)
        cut = max(1, len(paths) // 2)
        if len(paths) == 1:
            (valid if index % 2 == 0 else test).extend(paths)
        else:
            valid.extend(paths[:cut])
            test.extend(paths[cut:])
    return valid, test


def load_esc_entries(root, categories, kind):
    root = Path(root)
    audio_root = root / "audio"
    entries = {"train": [], "valid": [], "test": []}
    fold_to_split = {1: "train", 2: "train", 3: "train", 4: "valid", 5: "test"}
    with open(root / "meta" / "esc50.csv", "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] not in categories:
                continue
            split = fold_to_split[int(row["fold"])]
            entries[split].append(
                NoiseEntry(
                    audio_root / row["filename"],
                    kind,
                    row["category"],
                    f"esc:{row['src_file']}",
                )
            )
    return entries


def gather_noise_pools(args):
    pools = {split: defaultdict(list) for split in ["train", "valid", "test"]}

    ms_root = Path(args.ms_snsd_root)
    train_ms = [
        path
        for path in list_wavs(ms_root / "noise_train")
        if ms_category(path) in MS_SNSD_CATEGORIES
    ]
    test_ms = [
        path
        for path in list_wavs(ms_root / "noise_test")
        if ms_category(path) in MS_SNSD_CATEGORIES
    ]
    valid_ms, final_test_ms = split_files_by_category(test_ms, args.split_seed)
    for split, paths in {
        "train": train_ms,
        "valid": valid_ms,
        "test": final_test_ms,
    }.items():
        pools[split]["ms_snsd"] = [
            NoiseEntry(path, "ms_snsd", ms_category(path), f"ms:{path.name}")
            for path in paths
        ]

    presto_entries = []
    for root in [Path(args.presto_root), Path(args.pcafeter_root)]:
        presto_entries.extend(
            NoiseEntry(path, "presto_pcafeter", root.name.upper(), f"{root.name}:{path.name}")
            for path in list_wavs(root)
        )
    presto_splits = split_entries_by_group(
        presto_entries,
        args.split_seed + 2000,
        args.noise_valid_file_fraction,
        args.noise_test_file_fraction,
    )
    for split, entries in presto_splits.items():
        pools[split]["presto_pcafeter"] = entries

    rirs_root = Path(args.rirs_noises_root)
    parsed_noise = parse_list_entries(
        rirs_root / "real_rirs_isotropic_noises" / "noise_list",
        "room-linkage",
        rirs_root.parent,
        "rirs_isotropic",
    )
    isotropic = [
        NoiseEntry(path, source, room_id, f"isotropic:{room_id}")
        for path, room_id, source in parsed_noise
        if path.exists()
    ]
    isotropic_splits = split_entries_by_group(
        isotropic,
        args.split_seed + 3000,
        args.noise_valid_file_fraction,
        args.noise_test_file_fraction,
    )
    for split, entries in isotropic_splits.items():
        pools[split]["rirs_isotropic"] = entries

    esc_background = load_esc_entries(
        args.esc50_root, ESC_BACKGROUND_CATEGORIES, "esc_background"
    )
    for split, entries in esc_background.items():
        pools[split]["esc_background"] = entries

    events = load_esc_entries(args.esc50_root, ESC_EVENT_CATEGORIES, "esc_event")
    return pools, events


def prepare_rir(entry, args, rng):
    rir, channel = read_audio(entry.path, args.fs, rng=rng, select_channel=True)
    if not len(rir):
        raise ValueError(f"Empty RIR: {entry.path}")
    direct_index = int(np.argmax(np.abs(rir)))
    rir = rir / (np.sqrt(np.sum(rir.astype(np.float64) ** 2)) + 1e-8)
    return rir.astype(np.float32), direct_index, channel


def apply_rir(clean, entry, args, rng):
    rir, direct_index, channel = prepare_rir(entry, args, rng)
    reverberant = fftconvolve(clean, rir, mode="full").astype(np.float32)
    end = direct_index + len(clean)
    if len(reverberant) < end:
        reverberant = np.pad(reverberant, (0, end - len(reverberant)))
    reverberant = reverberant[direct_index:end]
    early = rir.copy()
    early_end = min(
        len(early),
        direct_index + int(round(args.early_reflections_ms * args.fs / 1000.0)),
    )
    early[early_end:] = 0.0
    target = fftconvolve(clean, early, mode="full").astype(np.float32)
    target = target[direct_index : direct_index + len(clean)]
    if len(target) < len(clean):
        target = np.pad(target, (0, len(clean) - len(target)))
    rt60, drr = estimate_rir_metrics(rir, direct_index, args.fs)
    return reverberant, target, direct_index, rt60, drr, channel


def sample_rir_application(clean, rir_pools, args, rng):
    for _ in range(max(1, args.rir_attempts)):
        source = weighted_source_choice(rir_pools, RIR_SOURCE_WEIGHTS, rng)
        entry = rng.choice(rir_pools[source])
        result = apply_rir(clean, entry, args, rng)
        rt60 = result[3]
        if not math.isnan(rt60) and args.rt60_min <= rt60 <= args.rt60_max:
            return source, entry, result
    raise RuntimeError(
        f"Could not sample an RIR with RT60 in [{args.rt60_min}, {args.rt60_max}] "
        f"after {args.rir_attempts} attempts"
    )


def prepare_background(entry, samples, args, rng):
    wav, _ = read_audio(entry.path, args.fs)
    wav, start = take_segment(wav, samples, rng)
    return (wav - float(np.mean(wav))).astype(np.float32), start


def prepare_event(entry, samples, args, rng):
    wav, _ = read_audio(entry.path, args.fs)
    wav = wav - float(np.mean(wav))
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak <= 1e-6:
        return np.zeros(samples, dtype=np.float32), 0, 0
    active = np.flatnonzero(np.abs(wav) >= peak * args.event_trim_threshold)
    if not len(active):
        return np.zeros(samples, dtype=np.float32), 0, 0
    margin = int(round(args.event_margin_seconds * args.fs))
    start = max(0, int(active[0]) - margin)
    end = min(len(wav), int(active[-1]) + margin + 1)
    event = wav[start:end]
    if len(event) > samples:
        crop = rng.randint(0, len(event) - samples)
        event = event[crop : crop + samples]
    offset = rng.randint(0, max(0, samples - len(event)))
    canvas = np.zeros(samples, dtype=np.float32)
    canvas[offset : offset + len(event)] = event
    return canvas, offset, len(event)


def scale_event(speech, event, event_length, snr_db, args):
    if event_length <= 0:
        return event
    active = event[np.abs(event) > 1e-8]
    if not len(active):
        return event
    target_rms = rms(speech) / (10.0 ** (snr_db / 20.0))
    scaled = event * (target_rms / (rms(active) + 1e-12))
    speech_peak = max(float(np.max(np.abs(speech))), 1e-6)
    event_peak = max(float(np.max(np.abs(scaled))), 1e-6)
    max_event_peak = speech_peak * args.event_peak_ratio
    if event_peak > max_event_peak:
        scaled *= max_event_peak / event_peak
    return scaled.astype(np.float32)


def relative_path(path, dataset_root):
    return os.path.relpath(path, dataset_root).replace("\\", "/")


def generate_split(split, count, speaker_groups, rir_pools, noise_pools, events, args, rng):
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
        scene_type = sample_scene_type("classroom_v2", rng)
        has_speech = scene_type != "noise_only"
        use_rir = scene_type in {"reverb_only", "reverb_noise"}
        use_noise = scene_type in {"reverb_noise", "noise_no_reverb", "noise_only"}

        if has_speech:
            clean, used_clean, activity, speaker = build_stitched_speech(
                speaker_groups, samples, args, rng
            )
        else:
            clean = np.zeros(samples, dtype=np.float32)
            used_clean, activity, speaker = [], 0.0, ""

        rir_entry = None
        rir_source = ""
        rir_channel = 0
        direct_index = 0
        rt60, drr = float("nan"), float("nan")
        if use_rir:
            rir_source, rir_entry, rir_result = sample_rir_application(
                clean, rir_pools, args, rng
            )
            reverberant, early_target, direct_index, rt60, drr, rir_channel = rir_result
            target = early_target
        else:
            reverberant, target = clean.copy(), clean.copy()

        noise_entry = None
        noise_source = ""
        noise_start = 0
        background_snr = float("nan")
        if use_noise:
            noise_source = weighted_source_choice(noise_pools, NOISE_SOURCE_WEIGHTS, rng)
            noise_entry = rng.choice(noise_pools[noise_source])
            background, noise_start = prepare_background(noise_entry, samples, args, rng)
            if has_speech:
                background_snr = sample_snr_db(args, rng)
                target_noise_rms = rms(reverberant) / (10.0 ** (background_snr / 20.0))
                background *= target_noise_rms / (rms(background) + 1e-12)
            else:
                background = scale_to_dbfs(
                    background,
                    rng.uniform(args.noise_only_dbfs_min, args.noise_only_dbfs_max),
                )
            noisy = reverberant + background
        else:
            noisy = reverberant.copy()

        event_entry = None
        event_offset = 0
        event_length = 0
        event_snr = float("nan")
        if has_speech and use_noise and events and rng.random() < args.event_probability:
            event_entry = rng.choice(events)
            event, event_offset, event_length = prepare_event(event_entry, samples, args, rng)
            event_snr = rng.uniform(args.event_snr_min, args.event_snr_max)
            noisy += scale_event(reverberant, event, event_length, event_snr, args)

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
                "target_mode": "early_reflections",
                "clean_files": "|".join(relative_path(path, args.dataset_root) for path, _, _ in used_clean),
                "clean_spans": "|".join(f"{start}:{length}" for _, start, length in used_clean),
                "speaker_id": speaker,
                "speech_activity": activity,
                "noise_file": "" if noise_entry is None else relative_path(noise_entry.path, args.dataset_root),
                "noise_source": noise_source,
                "noise_category": "" if noise_entry is None else noise_entry.category,
                "noise_start_sample": noise_start,
                "background_snr_db": "" if math.isnan(background_snr) else background_snr,
                "event_file": "" if event_entry is None else relative_path(event_entry.path, args.dataset_root),
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
    parser = argparse.ArgumentParser(description="Generate classroom_v4 paired speech data.")
    parser.add_argument("--dataset-root", default=r"..\dataset")
    parser.add_argument("--clean-root", default=r"..\dataset\clean_trainset_28spk_wav")
    parser.add_argument("--test-clean-root", default=r"..\dataset\clean_testset_wav")
    parser.add_argument("--clean-layout", choices=["auto", "voicebank"], default="voicebank")
    parser.add_argument("--valid-speaker-fraction", type=float, default=0.1)
    parser.add_argument("--but-rirs-root", default=r"..\dataset\BUT_ReverbDB_rel_19_06_RIR-Only")
    parser.add_argument("--rirs-noises-root", default=r"..\dataset\RIRS_NOISES")
    parser.add_argument("--ms-snsd-root", default=r"..\dataset\MS-SNSD-sparse")
    parser.add_argument("--esc50-root", default=r"..\dataset\ESC-50-master")
    parser.add_argument("--presto-root", default=r"..\dataset\PRESTO")
    parser.add_argument("--pcafeter-root", default=r"..\dataset\PCAFETER")
    parser.add_argument("--out-root", default=r"..\dataset_classroom_v4\generated")
    parser.add_argument("--num-train", type=int, default=20000)
    parser.add_argument("--num-valid", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--clean-dbfs-min", type=float, default=-28.0)
    parser.add_argument("--clean-dbfs-max", type=float, default=-18.0)
    parser.add_argument("--speech-activity-dbfs", type=float, default=-40.0)
    parser.add_argument("--min-speech-activity", type=float, default=0.4)
    parser.add_argument("--clean-file-attempts", type=int, default=10)
    parser.add_argument("--max-stitched-files", type=int, default=4)
    parser.add_argument("--gap-min-seconds", type=float, default=0.08)
    parser.add_argument("--gap-max-seconds", type=float, default=0.30)
    parser.add_argument("--early-reflections-ms", type=float, default=50.0)
    parser.add_argument("--rt60-min", type=float, default=0.15)
    parser.add_argument("--rt60-max", type=float, default=1.5)
    parser.add_argument("--rir-attempts", type=int, default=30)
    parser.add_argument("--snr-profile", choices=["quiet_classroom", "classroom", "uniform"], default="quiet_classroom")
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--noise-only-dbfs-min", type=float, default=-38.0)
    parser.add_argument("--noise-only-dbfs-max", type=float, default=-28.0)
    parser.add_argument("--event-probability", type=float, default=0.10)
    parser.add_argument("--event-snr-min", type=float, default=18.0)
    parser.add_argument("--event-snr-max", type=float, default=30.0)
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
    args.min_gap_samples = int(round(args.gap_min_seconds * args.fs))
    args.max_gap_samples = int(round(args.gap_max_seconds * args.fs))

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.rglob("*.wav")) and not args.overwrite:
        raise FileExistsError(f"{out_root} already contains wav files")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    clean_splits = choose_clean_files(args)
    speaker_groups = {
        split: group_clean_by_speaker(files) for split, files in clean_splits.items()
    }
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
    noise_pools, event_splits = gather_noise_pools(args)

    for split in ["train", "valid", "test"]:
        print(
            f"{split}: speakers={len(speaker_groups[split])}, "
            f"rirs={sum(len(v) for v in rir_pools[split].values())}, "
            f"noises={sum(len(v) for v in noise_pools[split].values())}, "
            f"events={len(event_splits[split])}"
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
            noise_pools[split],
            event_splits[split],
            args,
            rng,
        )

    config = vars(args).copy()
    config["rir_source_weights"] = RIR_SOURCE_WEIGHTS
    config["noise_source_weights"] = NOISE_SOURCE_WEIGHTS
    config["ms_snsd_categories"] = sorted(MS_SNSD_CATEGORIES)
    config["esc_background_categories"] = sorted(ESC_BACKGROUND_CATEGORIES)
    config["esc_event_categories"] = sorted(ESC_EVENT_CATEGORIES)
    config["rooms_by_split"] = {
        split: {source: rooms for source, rooms in sources.items()}
        for split, sources in room_splits.items()
    }
    config["rir_counts_by_split"] = {
        split: {source: len(entries) for source, entries in sources.items()}
        for split, sources in rir_pools.items()
    }
    config["noise_counts_by_split"] = {
        split: {source: len(entries) for source, entries in sources.items()}
        for split, sources in noise_pools.items()
    }
    config["event_counts_by_split"] = {
        split: len(entries) for split, entries in event_splits.items()
    }
    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()

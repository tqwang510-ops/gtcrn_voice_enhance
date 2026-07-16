import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


SPLITS = ("train", "valid", "test")


def read_metadata(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nonempty_values(rows, field):
    return {row[field] for row in rows if row.get(field)}


def overlap_count(left, right, field):
    return len(nonempty_values(left, field) & nonempty_values(right, field))


def numeric_summary(rows, field):
    values = sorted(float(row[field]) for row in rows if row.get(field) not in {None, ""})
    if not values:
        return None
    indexes = [0, len(values) // 4, len(values) // 2, 3 * len(values) // 4, -1]
    return {
        "count": len(values),
        "min": values[indexes[0]],
        "p25": values[indexes[1]],
        "median": values[indexes[2]],
        "p75": values[indexes[3]],
        "max": values[indexes[4]],
    }


def count_field(rows, field):
    return dict(sorted(Counter(row[field] for row in rows if row.get(field)).items()))


def audit_audio(root, expected_fs, expected_seconds, sample_count, seed):
    paths = [
        path
        for split in SPLITS
        for kind in ("clean", "noisy")
        for path in (root / split / kind).glob("*.wav")
    ]
    rng = random.Random(seed)
    selected = rng.sample(paths, min(sample_count, len(paths)))
    expected_frames = int(round(expected_fs * expected_seconds))
    bad_format = 0
    nonfinite = 0
    peak_over_limit = 0
    silent_noisy = 0
    max_peak = 0.0
    for path in selected:
        wav, fs = sf.read(path, dtype="float32", always_2d=True)
        if fs != expected_fs or wav.shape != (expected_frames, 1):
            bad_format += 1
        if not np.isfinite(wav).all():
            nonfinite += 1
        peak = float(np.max(np.abs(wav)))
        max_peak = max(max_peak, peak)
        if peak > 0.981:
            peak_over_limit += 1
        if path.parent.name == "noisy":
            level = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            if level < 1e-5:
                silent_noisy += 1
    return {
        "total_wav_files": len(paths),
        "unique_file_sizes": sorted({path.stat().st_size for path in paths}),
        "sampled_wav_files": len(selected),
        "bad_format_or_length": bad_format,
        "nonfinite_waveforms": nonfinite,
        "sampled_peak_over_0_981": peak_over_limit,
        "sampled_silent_noisy": silent_noisy,
        "max_sampled_peak": max_peak,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit a generated classroom dataset.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--audio-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    root = Path(args.root)
    metadata = {
        split: read_metadata(root / "metadata" / f"{split}.csv") for split in SPLITS
    }
    train, valid, test = (metadata[split] for split in SPLITS)
    report = {
        "root": str(root.resolve()),
        "counts": {split: len(rows) for split, rows in metadata.items()},
        "hours": {
            split: sum(float(row["segment_seconds"]) for row in rows) / 3600.0
            for split, rows in metadata.items()
        },
        "overlap": {
            field: {
                "train_valid": overlap_count(train, valid, field),
                "train_test": overlap_count(train, test, field),
                "valid_test": overlap_count(valid, test, field),
            }
            for field in ("speaker_id", "room_id", "noise_file", "event_file")
        },
        "train_distribution": {
            field: count_field(train, field)
            for field in ("scene_type", "rir_source", "noise_source", "event_category")
        },
        "train_numeric": {
            "speech_activity": numeric_summary(
                [row for row in train if row.get("speaker_id")], "speech_activity"
            ),
            "clean_dbfs": numeric_summary(
                [row for row in train if row.get("speaker_id")], "clean_dbfs"
            ),
            **{
                field: numeric_summary(train, field)
                for field in (
                    "background_snr_db",
                    "event_snr_db",
                    "rt60_estimate_s",
                    "drr_db",
                    "noisy_dbfs",
                    "final_gain",
                )
            },
        },
        "train_unique": {
            field: len(nonempty_values(train, field))
            for field in ("speaker_id", "room_id", "rir_file", "noise_file", "event_file")
        },
        "audio": audit_audio(
            root, args.fs, args.segment_seconds, args.audio_samples, args.seed
        ),
    }
    report["passed"] = (
        all(
            count == 0
            for field_counts in report["overlap"].values()
            for count in field_counts.values()
        )
        and report["audio"]["bad_format_or_length"] == 0
        and report["audio"]["nonfinite_waveforms"] == 0
        and report["audio"]["sampled_peak_over_0_981"] == 0
        and report["audio"]["sampled_silent_noisy"] == 0
        and (
            math.isclose(
                report["train_numeric"]["speech_activity"]["min"],
                0.4,
                abs_tol=0.01,
            )
            or report["train_numeric"]["speech_activity"]["min"] > 0.4
        )
    )
    output = root / "metadata" / "audit.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

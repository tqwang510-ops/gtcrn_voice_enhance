"""Layer low-level PRESTO/PCAFETER student murmur over continuous classroom data."""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(wav):
    return float(np.sqrt(np.mean(np.square(wav), dtype=np.float64) + 1e-12))


def source_files(root):
    files = sorted(Path(root).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {root}")
    return files


def choose_murmur(pools, split, samples, rng, train_end, valid_end):
    source_name = rng.choice(("presto", "pcafeter"))
    path = rng.choice(pools[source_name])
    wav, fs = sf.read(path, dtype="float32")
    if fs != 16000 or wav.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz WAV: {path}")
    split_bounds = {
        "train": (0.0, train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, 1.0),
    }
    start_fraction, end_fraction = split_bounds[split]
    first = int(len(wav) * start_fraction)
    last = int(len(wav) * end_fraction) - samples
    if last < first:
        raise ValueError(f"Murmur split is shorter than one segment: {path} {split}")
    start = rng.randint(first, last)
    return wav[start : start + samples].astype(np.float32), path, start, source_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--presto-root", default=r"..\dataset\PRESTO")
    parser.add_argument("--pcafeter-root", default=r"..\dataset\PCAFETER")
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--snr-main-fraction", type=float, default=0.75)
    parser.add_argument("--snr-main-min", type=float, default=15.0)
    parser.add_argument("--snr-main-max", type=float, default=24.0)
    parser.add_argument("--snr-low-min", type=float, default=10.0)
    parser.add_argument("--snr-low-max", type=float, default=15.0)
    parser.add_argument("--train-time-end", type=float, default=0.70)
    parser.add_argument("--valid-time-end", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if not 0.0 < args.train_time_end < args.valid_time_end < 1.0:
        raise ValueError("Expected 0 < train-time-end < valid-time-end < 1")
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.rglob("*.wav")) and not args.overwrite:
        raise FileExistsError(f"{out_root} already contains WAV files")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    pools = {
        "presto": source_files(args.presto_root),
        "pcafeter": source_files(args.pcafeter_root),
    }
    rng = random.Random(args.seed)
    summary = {"files": 0, "murmur_files": 0, "by_source": {}}
    for split in ("train", "valid", "test"):
        in_meta = Path(args.input_root) / "metadata" / f"{split}.csv"
        out_meta = out_root / "metadata" / f"{split}.csv"
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(in_meta, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        output_rows = []
        for row in rows:
            name = row["file"]
            clean_path = Path(args.input_root) / split / "clean" / name
            noisy_path = Path(args.input_root) / split / "noisy" / name
            out_clean = out_root / split / "clean" / name
            out_noisy = out_root / split / "noisy" / name
            out_clean.parent.mkdir(parents=True, exist_ok=True)
            out_noisy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clean_path, out_clean)
            noisy, fs = sf.read(noisy_path, dtype="float32")
            if fs != 16000 or noisy.ndim != 1:
                raise ValueError(f"Expected mono 16 kHz WAV: {noisy_path}")
            murmur_path = ""
            murmur_snr = ""
            murmur_start = ""
            murmur_source = ""
            if row.get("scene_type") not in {"identity", "noise_only"} and rng.random() < args.fraction:
                murmur, source_path, source_start, source_name = choose_murmur(
                    pools,
                    split,
                    len(noisy),
                    rng,
                    args.train_time_end,
                    args.valid_time_end,
                )
                if rng.random() < args.snr_main_fraction:
                    snr = rng.uniform(args.snr_main_min, args.snr_main_max)
                else:
                    snr = rng.uniform(args.snr_low_min, args.snr_low_max)
                target_rms = rms(noisy) / (10.0 ** (snr / 20.0))
                noisy = noisy + murmur * (target_rms / rms(murmur))
                murmur_path = str(source_path)
                murmur_snr = f"{snr:.8f}"
                murmur_start = str(source_start)
                murmur_source = source_name
                summary["murmur_files"] += 1
                source_name = source_path.parent.name
                summary["by_source"][source_name] = summary["by_source"].get(source_name, 0) + 1
            peak = float(np.max(np.abs(noisy)))
            if peak > 0.98:
                noisy = noisy * (0.98 / peak)
            sf.write(out_noisy, noisy.astype(np.float32), fs, subtype="PCM_16")
            row["student_murmur_file"] = murmur_path
            row["student_murmur_snr_db"] = murmur_snr
            row["student_murmur_start_sample"] = murmur_start
            row["student_murmur_source"] = murmur_source
            output_rows.append(row)
            summary["files"] += 1
        with open(out_meta, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
        with open(out_root / "metadata" / f"{split}.json", "w", encoding="utf-8") as handle:
            json.dump({"files": [row["file"] for row in output_rows]}, handle, indent=2)
    with open(out_root / "metadata" / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args) | summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

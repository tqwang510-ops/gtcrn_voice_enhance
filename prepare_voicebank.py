import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SPLITS = (
    ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav", "train"),
    ("clean_testset_wav", "noisy_testset_wav", "valid"),
)


def wav_files(directory):
    files = list(Path(directory).rglob("*.wav"))
    by_name = {path.name: path for path in files}
    if len(by_name) != len(files):
        raise ValueError(f"Duplicate wav names found under {directory}")
    return by_name


def prepare_file(source, destination, source_fs, target_fs, overwrite):
    if destination.exists() and not overwrite:
        return "skipped"

    wav, fs = sf.read(source, dtype="float32", always_2d=True)
    if fs != source_fs:
        raise ValueError(f"{source} sample rate is {fs}, expected {source_fs}")

    gcd = math.gcd(source_fs, target_fs)
    wav = resample_poly(wav, target_fs // gcd, source_fs // gcd, axis=0).astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    sf.write(temporary, wav, target_fs, subtype="PCM_16")
    os.replace(temporary, destination)
    return "written"


def build_pairs(dataset_root):
    pairs = []
    for clean_source, noisy_source, split in SPLITS:
        clean = wav_files(dataset_root / clean_source)
        noisy = wav_files(dataset_root / noisy_source)
        if clean.keys() != noisy.keys():
            missing_clean = sorted(noisy.keys() - clean.keys())[:5]
            missing_noisy = sorted(clean.keys() - noisy.keys())[:5]
            raise ValueError(
                f"Unpaired files in {split}: missing clean={missing_clean}, "
                f"missing noisy={missing_noisy}"
            )
        for name in sorted(clean):
            pairs.append((noisy[name], clean[name], split, name))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VoiceBank-DEMAND 28spk wav files for custom GTCRN training."
    )
    parser.add_argument("--dataset-root", default="../dataset")
    parser.add_argument("--source-fs", type=int, default=48000)
    parser.add_argument("--target-fs", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    pairs = build_pairs(dataset_root)
    if args.max_files:
        pairs = pairs[:args.max_files]

    print(f"paired files: {len(pairs)}")
    print(f"resample: {args.source_fs} Hz -> {args.target_fs} Hz")
    if args.dry_run:
        print("dry run complete")
        return

    jobs = []
    for noisy, clean, split, name in pairs:
        jobs.append((noisy, dataset_root / split / "noisy" / name))
        jobs.append((clean, dataset_root / split / "clean" / name))

    written = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                prepare_file,
                source,
                destination,
                args.source_fs,
                args.target_fs,
                args.overwrite,
            )
            for source, destination in jobs
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            written += result == "written"
            skipped += result == "skipped"
            if index % 100 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)} files, written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()

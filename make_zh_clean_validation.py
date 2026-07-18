import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from make_ceiling_pa_dataset import rms_dbfs, scale_to_dbfs
from make_classroom_v5_chinese_dataset import (
    gather_aishell_splits,
    load_transcript_ids,
)
from make_classroom_v4_dataset import read_audio


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build paired identity validation sets (noisy == clean) from AISHELL "
            "dev speakers at native (raw) and normalized levels."
        )
    )
    parser.add_argument("--aishell-root", default=r"..\dataset\data_aishell\wav_extracted")
    parser.add_argument(
        "--transcript",
        default=r"..\dataset\data_aishell\transcript\aishell_transcript_v0.8.txt",
    )
    parser.add_argument("--out-root", default=r"..\dataset_classroom_v5_chinese\zh_clean_valid")
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--speakers", type=int, default=20)
    parser.add_argument("--per-speaker", type=int, default=3)
    parser.add_argument("--normalized-dbfs", type=float, default=-25.0)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    samples = int(round(args.segment_seconds * args.fs))
    rng = random.Random(args.seed)
    transcript_ids = load_transcript_ids(args.transcript)
    speaker_groups, _ = gather_aishell_splits(args.aishell_root, transcript_ids)
    dev_speakers = sorted(speaker_groups["valid"])
    speakers = dev_speakers[: args.speakers]

    rows = []
    index = 0
    for speaker in speakers:
        files = speaker_groups["valid"][speaker].copy()
        rng.shuffle(files)
        for path in files[: args.per_speaker]:
            wav, _ = read_audio(path, args.fs)
            wav = wav - float(np.mean(wav))
            if len(wav) >= samples:
                start = rng.randint(0, len(wav) - samples)
                segment = wav[start : start + samples]
            else:
                segment = np.pad(wav, (0, samples - len(wav)))
            native_dbfs = rms_dbfs(segment)
            name = f"zhclean_{index:04d}.wav"
            variants = {
                "raw": segment.astype(np.float32),
                "normalized": scale_to_dbfs(segment, args.normalized_dbfs),
            }
            for variant, audio in variants.items():
                for kind in ["clean", "noisy"]:
                    out_dir = Path(args.out_root) / variant / kind
                    out_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(out_dir / name, audio, args.fs, subtype="PCM_16")
            rows.append(
                {
                    "file": name,
                    "speaker_id": speaker,
                    "source_file": str(path),
                    "native_dbfs": native_dbfs,
                    "raw_dbfs": rms_dbfs(variants["raw"]),
                    "normalized_dbfs": rms_dbfs(variants["normalized"]),
                    "segment_seconds": args.segment_seconds,
                    "fs": args.fs,
                }
            )
            index += 1

    metadata_dir = Path(args.out_root) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "files.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(metadata_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "files": len(rows),
                "speakers": speakers,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(f"wrote {index} identity pairs (raw + normalized) to {args.out_root}")


if __name__ == "__main__":
    main()

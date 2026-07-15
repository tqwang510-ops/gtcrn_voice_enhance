import argparse
import json
import math
import random
from pathlib import Path


def speaker_id(file_name):
    return file_name.split("_", 1)[0]


def save_manifest(path, files, speakers, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **metadata,
        "speakers": sorted(speakers),
        "num_files": len(files),
        "files": sorted(files),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible speaker-disjoint VoiceBank manifests."
    )
    parser.add_argument("--dataset-root", default="../dataset")
    parser.add_argument("--output-dir", default="../dataset/splits/voicebank_serious")
    parser.add_argument("--valid-speaker-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    train_names = sorted(path.name for path in (dataset_root / "train" / "clean").glob("*.wav"))
    test_names = sorted(path.name for path in (dataset_root / "valid" / "clean").glob("*.wav"))
    if not train_names or not test_names:
        raise ValueError("Prepared train and official test wav files are required")

    speakers = sorted({speaker_id(name) for name in train_names})
    shuffled = speakers.copy()
    random.Random(args.seed).shuffle(shuffled)
    valid_count = max(1, math.ceil(len(speakers) * args.valid_speaker_fraction))
    valid_speakers = set(shuffled[:valid_count])
    train_speakers = set(speakers) - valid_speakers

    train_files = [name for name in train_names if speaker_id(name) in train_speakers]
    valid_files = [name for name in train_names if speaker_id(name) in valid_speakers]
    test_speakers = {speaker_id(name) for name in test_names}
    metadata = {
        "dataset": "VoiceBank-DEMAND 28spk",
        "seed": args.seed,
        "valid_speaker_fraction": args.valid_speaker_fraction,
    }

    save_manifest(
        output_dir / "train.json", train_files, train_speakers, {**metadata, "split": "train"}
    )
    save_manifest(
        output_dir / "valid.json", valid_files, valid_speakers, {**metadata, "split": "valid"}
    )
    save_manifest(
        output_dir / "test.json", test_names, test_speakers, {**metadata, "split": "test"}
    )

    print(f"train speakers: {sorted(train_speakers)}")
    print(f"valid speakers: {sorted(valid_speakers)}")
    print(f"train files: {len(train_files)}")
    print(f"valid files: {len(valid_files)}")
    print(f"test files: {len(test_names)}")


if __name__ == "__main__":
    main()

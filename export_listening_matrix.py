"""Export listening matrix triples (noisy/enhanced/clean) for picked v5 test files.

Reads a manifest JSON produced during Step 7 (list of {file, tag, ...}) and writes
<tag>__<stem>_{noisy,enhanced,clean}.wav into the output directory.
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from audio_utils import enhance_waveform, read_wav
from gtcrn import GTCRN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    fs = config.get("fs", 16000)
    model = GTCRN(nfft=config.get("n_fft", 256), fs=fs).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    entries = json.load(open(args.manifest, "r", encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir = Path(args.noisy_dir)
    clean_dir = Path(args.clean_dir)

    for entry in entries:
        name = entry["file"]
        tag = entry["tag"]
        stem = Path(name).stem
        noisy, _ = read_wav(noisy_dir / name)
        clean, _ = read_wav(clean_dir / name)
        length = min(len(noisy), len(clean))
        noisy, clean = noisy[:length], clean[:length]
        wav = torch.from_numpy(noisy).to(device)
        with torch.no_grad():
            enhanced = (
                enhance_waveform(
                    model,
                    wav,
                    config.get("n_fft", 256),
                    config.get("hop_length", 80),
                    config.get("win_length", 160),
                    center=config.get("center", True),
                )
                .detach()
                .cpu()
                .numpy()
            )
        sf.write(out_dir / f"{tag}__{stem}_noisy.wav", noisy, fs)
        sf.write(out_dir / f"{tag}__{stem}_enhanced.wav", enhanced, fs)
        sf.write(out_dir / f"{tag}__{stem}_clean.wav", clean, fs)
        print(f"saved {tag}__{stem}")
    print(f"done: {len(entries)} files -> {out_dir}")


if __name__ == "__main__":
    main()

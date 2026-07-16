import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly

from audio_utils import enhance_waveform, read_wav, rms_dbfs, si_snr_db
from gtcrn import GTCRN


def read_wav_for_eval(path, expected_fs):
    wav, fs = read_wav(path)
    if fs == expected_fs:
        return wav, fs
    divisor = math.gcd(fs, expected_fs)
    wav = resample_poly(wav, expected_fs // divisor, fs // divisor).astype(np.float32)
    return wav, expected_fs


def load_manifest(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    return data.get("files", data) if isinstance(data, dict) else data


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


METRIC_FIELDS = [
    "file",
    "scene_type",
    "seconds",
    "clean_rms_dbfs",
    "input_rms_dbfs",
    "enhanced_rms_dbfs",
    "noise_attenuation_db",
    "input_si_snr_db",
    "enhanced_si_snr_db",
    "si_snr_improvement_db",
    "input_pesq_wb",
    "enhanced_pesq_wb",
    "pesq_improvement",
    "input_stoi",
    "enhanced_stoi",
    "stoi_improvement",
    "error",
]


def save_sample_group(
    rows,
    samples_dir,
    noisy_dir,
    clean_dir,
    model,
    device,
    fs,
    n_fft,
    hop_length,
    win_length,
    center,
):
    samples_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row["file"]
        noisy, _ = read_wav_for_eval(noisy_dir / name, fs)
        clean, _ = read_wav_for_eval(clean_dir / name, fs)
        wav = torch.from_numpy(noisy).to(device)
        with torch.no_grad():
            enhanced = enhance_waveform(
                model, wav, n_fft, hop_length, win_length, center=center
            ).detach().cpu().numpy()
        stem = Path(name).stem
        sf.write(samples_dir / f"{stem}_noisy.wav", noisy, fs)
        sf.write(samples_dir / f"{stem}_enhanced.wav", enhanced, fs)
        sf.write(samples_dir / f"{stem}_clean.wav", clean, fs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GTCRN checkpoint on paired wav files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--metadata-csv",
        default="",
        help="Optional scene metadata with file and scene_type columns.",
    )
    parser.add_argument("--out-dir", default="evaluation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--save-hardest", type=int, default=10)
    parser.add_argument(
        "--save-worst",
        type=int,
        default=-1,
        help="Save samples with the lowest SI-SNR improvement; -1 uses --save-hardest.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    fs = config.get("fs", 16000)
    win_length = config.get("win_length", 160)
    hop_length = config.get("hop_length", 80)
    n_fft = config.get("n_fft", 256)
    center = config.get("center", True)

    model = GTCRN(nfft=n_fft, fs=fs).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    noisy_dir = Path(args.noisy_dir)
    clean_dir = Path(args.clean_dir)
    names = load_manifest(args.manifest)
    if names is None:
        names = sorted(path.name for path in noisy_dir.glob("*.wav"))
    if args.start_index:
        names = names[args.start_index :]
    if args.max_files:
        names = names[: args.max_files]
    if not names:
        raise ValueError("No evaluation wav files found")

    scene_types = {}
    if args.metadata_csv:
        with open(args.metadata_csv, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                scene_types[row["file"]] = row.get("scene_type", "")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    rows = []
    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for index, name in enumerate(names, start=1):
            try:
                noisy, noisy_fs = read_wav_for_eval(noisy_dir / name, fs)
                clean, clean_fs = read_wav_for_eval(clean_dir / name, fs)
                length = min(len(noisy), len(clean))
                noisy = noisy[:length]
                clean = clean[:length]
                wav = torch.from_numpy(noisy).to(device)
                with torch.no_grad():
                    enhanced = enhance_waveform(
                        model, wav, n_fft, hop_length, win_length, center=center
                    ).detach().cpu().numpy()
                enhanced = enhanced[:length]

                clean_level = rms_dbfs(clean)
                input_level = rms_dbfs(noisy)
                enhanced_level = rms_dbfs(enhanced)
                common = {
                    "file": name,
                    "scene_type": scene_types.get(name, ""),
                    "seconds": length / fs,
                    "clean_rms_dbfs": clean_level,
                    "input_rms_dbfs": input_level,
                    "enhanced_rms_dbfs": enhanced_level,
                    "error": "",
                }
                if clean_level < -50.0:
                    row = {
                        **common,
                        "noise_attenuation_db": input_level - enhanced_level,
                        "input_si_snr_db": "",
                        "enhanced_si_snr_db": "",
                        "si_snr_improvement_db": "",
                        "input_pesq_wb": "",
                        "enhanced_pesq_wb": "",
                        "pesq_improvement": "",
                        "input_stoi": "",
                        "enhanced_stoi": "",
                        "stoi_improvement": "",
                    }
                else:
                    input_score = si_snr_db(noisy, clean)
                    enhanced_score = si_snr_db(enhanced, clean)
                    input_pesq = float(pesq(fs, clean, noisy, "wb"))
                    enhanced_pesq = float(pesq(fs, clean, enhanced, "wb"))
                    input_stoi = float(stoi(clean, noisy, fs, extended=False))
                    enhanced_stoi = float(stoi(clean, enhanced, fs, extended=False))
                    row = {
                        **common,
                        "noise_attenuation_db": "",
                        "input_si_snr_db": input_score,
                        "enhanced_si_snr_db": enhanced_score,
                        "si_snr_improvement_db": enhanced_score - input_score,
                        "input_pesq_wb": input_pesq,
                        "enhanced_pesq_wb": enhanced_pesq,
                        "pesq_improvement": enhanced_pesq - input_pesq,
                        "input_stoi": input_stoi,
                        "enhanced_stoi": enhanced_stoi,
                        "stoi_improvement": enhanced_stoi - input_stoi,
                    }
            except Exception as exc:
                row = {
                    "file": name,
                    "scene_type": scene_types.get(name, ""),
                    "seconds": "",
                    "clean_rms_dbfs": "",
                    "input_rms_dbfs": "",
                    "enhanced_rms_dbfs": "",
                    "noise_attenuation_db": "",
                    "input_si_snr_db": "",
                    "enhanced_si_snr_db": "",
                    "si_snr_improvement_db": "",
                    "input_pesq_wb": "",
                    "enhanced_pesq_wb": "",
                    "pesq_improvement": "",
                    "input_stoi": "",
                    "enhanced_stoi": "",
                    "stoi_improvement": "",
                    "error": repr(exc),
                }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            if index % 50 == 0 or index == len(names):
                print(f"evaluated {index}/{len(names)}")

    successful_rows = [row for row in rows if not row["error"]]
    if args.metadata_csv:
        valid_rows = [
            row
            for row in successful_rows
            if row["scene_type"] not in {"clean", "noise_only"}
            and row["clean_rms_dbfs"] >= -50.0
        ]
        noise_only_rows = [
            row for row in successful_rows if row["scene_type"] == "noise_only"
        ]
        clean_rows = [row for row in successful_rows if row["scene_type"] == "clean"]
    else:
        valid_rows = [
            row for row in successful_rows if row["clean_rms_dbfs"] >= -50.0
        ]
        noise_only_rows = [
            row for row in successful_rows if row["clean_rms_dbfs"] < -50.0
        ]
        clean_rows = []
    if not valid_rows:
        raise ValueError("No valid speech rows available for summary metrics")
    improvements = np.array([row["si_snr_improvement_db"] for row in valid_rows])
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metadata_csv": str(Path(args.metadata_csv).resolve()) if args.metadata_csv else None,
        "excluded_speech_scenes": ["clean", "noise_only"] if args.metadata_csv else [],
        "files": len(rows),
        "files_used_for_summary": len(valid_rows),
        "speech_files": len(valid_rows),
        "noise_only_files": len(noise_only_rows),
        "clean_passthrough_files": len(clean_rows),
        "mean_input_si_snr_db": float(np.mean([row["input_si_snr_db"] for row in valid_rows])),
        "mean_enhanced_si_snr_db": float(
            np.mean([row["enhanced_si_snr_db"] for row in valid_rows])
        ),
        "mean_si_snr_improvement_db": float(np.mean(improvements)),
        "median_si_snr_improvement_db": float(np.median(improvements)),
        "improved_file_fraction": float(np.mean(improvements > 0.0)),
        "degraded_file_fraction": float(np.mean(improvements < 0.0)),
        "mean_input_pesq_wb": float(np.mean([row["input_pesq_wb"] for row in valid_rows])),
        "mean_enhanced_pesq_wb": float(
            np.mean([row["enhanced_pesq_wb"] for row in valid_rows])
        ),
        "mean_pesq_improvement": float(
            np.mean([row["pesq_improvement"] for row in valid_rows])
        ),
        "mean_input_stoi": float(np.mean([row["input_stoi"] for row in valid_rows])),
        "mean_enhanced_stoi": float(
            np.mean([row["enhanced_stoi"] for row in valid_rows])
        ),
        "mean_stoi_improvement": float(
            np.mean([row["stoi_improvement"] for row in valid_rows])
        ),
        "mean_noise_only_input_rms_dbfs": (
            float(np.mean([row["input_rms_dbfs"] for row in noise_only_rows]))
            if noise_only_rows
            else None
        ),
        "mean_noise_only_enhanced_rms_dbfs": (
            float(np.mean([row["enhanced_rms_dbfs"] for row in noise_only_rows]))
            if noise_only_rows
            else None
        ),
        "mean_noise_attenuation_db": (
            float(np.mean([row["noise_attenuation_db"] for row in noise_only_rows]))
            if noise_only_rows
            else None
        ),
        "mean_clean_enhanced_si_snr_db": (
            float(np.mean([row["enhanced_si_snr_db"] for row in clean_rows]))
            if clean_rows
            else None
        ),
        "mean_clean_pesq_improvement": (
            float(np.mean([row["pesq_improvement"] for row in clean_rows]))
            if clean_rows
            else None
        ),
        "mean_clean_stoi_improvement": (
            float(np.mean([row["stoi_improvement"] for row in clean_rows]))
            if clean_rows
            else None
        ),
        "config": config,
    }
    save_json(out_dir / "summary.json", summary)

    hardest = sorted(valid_rows, key=lambda row: row["input_si_snr_db"])[
        : args.save_hardest
    ]
    save_sample_group(
        hardest,
        out_dir / "hardest_samples",
        noisy_dir,
        clean_dir,
        model,
        device,
        fs,
        n_fft,
        hop_length,
        win_length,
        center,
    )

    worst_count = args.save_hardest if args.save_worst < 0 else args.save_worst
    worst = sorted(valid_rows, key=lambda row: row["si_snr_improvement_db"])[
        :worst_count
    ]
    save_sample_group(
        worst,
        out_dir / "worst_improvements",
        noisy_dir,
        clean_dir,
        model,
        device,
        fs,
        n_fft,
        hop_length,
        win_length,
        center,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

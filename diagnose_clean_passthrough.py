"""Diagnose clean-speech passthrough distortion for GTCRN checkpoints.

Feeds clean-scene files (input == target clean speech) through one or more
checkpoints and measures what the model does when it should do nothing:

- per-file SI-SNR / PESQ-WB / STOI against the clean reference, plus the same
  metrics for the unprocessed input so change values can be computed
- output-end equivalent transfer function M_eff = stft(enhanced) / stft(input)
  (this goes through the ISTFT/re-STFT round trip, so it is NOT an exact
  recovery of the model's internal mask, but it is valid for band/frame
  attenuation analysis)
- band energy gain: 0-1 kHz, 1-4 kHz, 4-8 kHz
- frame attenuation vs frame energy (voiced vs low-energy frames)
- attenuation on transient (onset) frames vs steady frames
- correlation of per-file transparency with speech_activity and loudness

Outputs metrics.csv, analysis.json, plots/*.png and worst spectrograms.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from scipy.stats import pearsonr, spearmanr

from audio_utils import enhance_waveform, read_wav, rms_dbfs, si_snr_db, sqrt_hann_window
from gtcrn import GTCRN

BANDS = [(0, 1000), (1000, 4000), (4000, 8000)]
BAND_NAMES = ["0-1kHz", "1-4kHz", "4-8kHz"]


def complex_stft(wav, n_fft, hop_length, win_length, center=True):
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    spec = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=sqrt_hann_window(win_length, tensor.device),
        center=center,
        return_complex=True,
    )
    return spec.numpy()  # (F, T) complex


def band_bin_range(fs, n_fft, low, high):
    bin_hz = fs / n_fft
    lo = max(1, int(math.ceil(low / bin_hz)))
    hi = min(n_fft // 2, int(math.floor(high / bin_hz)))
    return lo, hi


def analyze_file(input_wav, enhanced_wav, clean_wav, fs, n_fft, hop_length, win_length, center=True):
    """Return per-file metrics and frame/mask statistics for one model."""
    length = min(len(input_wav), len(enhanced_wav), len(clean_wav))
    input_wav = input_wav[:length]
    enhanced_wav = enhanced_wav[:length]
    clean_wav = clean_wav[:length]

    input_si_snr = si_snr_db(input_wav, clean_wav)
    input_pesq = float(pesq(fs, clean_wav, input_wav, "wb"))
    input_stoi = float(stoi(clean_wav, input_wav, fs, extended=False))
    result = {
        "si_snr_db": si_snr_db(enhanced_wav, clean_wav),
        "pesq_wb": float(pesq(fs, clean_wav, enhanced_wav, "wb")),
        "stoi": float(stoi(clean_wav, enhanced_wav, fs, extended=False)),
        "input_si_snr_db": input_si_snr,
        "input_pesq_wb": input_pesq,
        "input_stoi": input_stoi,
        "overall_gain_db": rms_dbfs(enhanced_wav) - rms_dbfs(input_wav),
    }
    result["si_snr_change_db"] = result["si_snr_db"] - input_si_snr
    result["pesq_change"] = result["pesq_wb"] - input_pesq
    result["stoi_change"] = result["stoi"] - input_stoi

    x_spec = complex_stft(input_wav, n_fft, hop_length, win_length, center=center)
    y_spec = complex_stft(enhanced_wav, n_fft, hop_length, win_length, center=center)
    x_mag = np.abs(x_spec)
    y_mag = np.abs(y_spec)
    x_energy = np.sum(x_mag * x_mag, axis=0)  # (T,)
    y_energy = np.sum(y_mag * y_mag, axis=0)

    peak = float(np.max(x_energy)) + 1e-20
    # Frames with any meaningful signal; below this the STFT is numerical noise.
    valid_frame = x_energy >= peak * 1e-6
    # Valid bins for mask division: within 80 dB of the file's spectral peak.
    spec_peak = float(np.max(x_mag)) + 1e-20
    valid_bin = x_mag >= spec_peak * 1e-4

    # Per-frame attenuation (dB) on valid frames.
    att = np.full(x_energy.shape, np.nan)
    att[valid_frame] = 10.0 * np.log10(
        (y_energy[valid_frame] + 1e-20) / (x_energy[valid_frame] + 1e-20)
    )

    frame_db = 10.0 * np.log10(x_energy / peak + 1e-20)
    voiced = valid_frame & (frame_db >= -20.0)
    low_energy = valid_frame & (frame_db < -20.0) & (frame_db >= -50.0)

    # Transient frames: top 10% positive spectral flux within voiced frames.
    flux = np.zeros(x_mag.shape[1])
    diff = x_mag[:, 1:] - x_mag[:, :-1]
    flux[1:] = np.sum(np.maximum(diff, 0.0), axis=0)
    voiced_flux = flux[voiced]
    if voiced_flux.size >= 10:
        transient_thr = np.quantile(voiced_flux, 0.9)
        steady_thr = np.quantile(voiced_flux, 0.5)
        transient = voiced & (flux >= transient_thr)
        steady = voiced & (flux <= steady_thr)
    else:
        transient = np.zeros_like(voiced)
        steady = np.zeros_like(voiced)

    def mean_att(mask):
        values = att[mask]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else None

    result["att_voiced_db"] = mean_att(voiced)
    result["att_low_energy_db"] = mean_att(low_energy)
    result["att_transient_db"] = mean_att(transient)
    result["att_steady_db"] = mean_att(steady)

    # Band gains and equivalent-transfer-function statistics (energy-weighted,
    # valid bins only).
    band_gains = []
    tf_mag_db = []
    tf_phase_deg = []
    transfer = np.zeros_like(x_spec, dtype=np.complex128)
    transfer[valid_bin] = y_spec[valid_bin] / x_spec[valid_bin]
    for low, high in BANDS:
        lo, hi = band_bin_range(fs, n_fft, low, high)
        xb = x_mag[lo : hi + 1, :]
        yb = y_mag[lo : hi + 1, :]
        vb = valid_bin[lo : hi + 1, :] & voiced[None, :]
        band_gains.append(
            float(10.0 * np.log10(np.sum(yb[vb] ** 2) / (np.sum(xb[vb] ** 2) + 1e-20)))
            if np.any(vb)
            else None
        )
        mb = transfer[lo : hi + 1, :]
        if np.any(vb):
            mags = np.abs(mb[vb])
            phases = np.abs(np.angle(mb[vb]))
            weights = xb[vb] ** 2
            tf_mag_db.append(float(np.average(20.0 * np.log10(mags + 1e-10), weights=weights)))
            tf_phase_deg.append(
                float(np.degrees(np.sqrt(np.average(phases * phases, weights=weights))))
            )
        else:
            tf_mag_db.append(None)
            tf_phase_deg.append(None)
    result["band_gain_db"] = dict(zip(BAND_NAMES, band_gains))
    result["transfer_mag_db"] = dict(zip(BAND_NAMES, tf_mag_db))
    result["transfer_phase_deg"] = dict(zip(BAND_NAMES, tf_phase_deg))

    # Frame-level records for scatter plots (subsampled later).
    frames = {
        "frame_db": frame_db[valid_frame],
        "att_db": att[valid_frame],
    }
    return result, frames


def plot_band_gains(per_model_results, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(len(BAND_NAMES))
    for index, (label, results) in enumerate(per_model_results.items()):
        means = [
            np.nanmean([r["band_gain_db"][b] for r in results if r["band_gain_db"][b] is not None])
            for b in BAND_NAMES
        ]
        ax.bar(x + (index - 1) * width, means, width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("band energy gain (dB)")
    ax.set_title("Clean passthrough: band energy gain by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_frame_scatter(per_model_frames, out_path, rng):
    fig, axes = plt.subplots(1, len(per_model_frames), figsize=(5 * len(per_model_frames), 4), sharey=True)
    if len(per_model_frames) == 1:
        axes = [axes]
    for ax, (label, frames) in zip(axes, per_model_frames.items()):
        xs = np.concatenate([f["frame_db"] for f in frames])
        ys = np.concatenate([f["att_db"] for f in frames])
        if xs.size > 4000:
            pick = rng.choice(xs.size, 4000, replace=False)
            xs, ys = xs[pick], ys[pick]
        ax.scatter(xs, ys, s=2, alpha=0.3)
        ax.axhline(0.0, color="red", linewidth=0.8)
        ax.axvline(-20.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("frame energy (dB below file peak)")
        ax.set_title(label)
    axes[0].set_ylabel("frame attenuation (dB)")
    fig.suptitle("Clean passthrough: attenuation vs frame energy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sisnr_hist(per_model_results, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, results in per_model_results.items():
        values = [r["si_snr_db"] for r in results]
        ax.hist(values, bins=20, alpha=0.5, label=f"{label} (mean {np.mean(values):.1f} dB)")
    ax.set_xlabel("enhanced SI-SNR vs clean (dB)")
    ax.set_ylabel("files")
    ax.set_title("Clean passthrough: per-file SI-SNR distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation(per_model_results, meta, key, xlabel, out_path):
    fig, axes = plt.subplots(1, len(per_model_results), figsize=(5 * len(per_model_results), 4), sharey=True)
    if len(per_model_results) == 1:
        axes = [axes]
    for ax, (label, results) in zip(axes, per_model_results.items()):
        xs = np.array([meta[r["file"]][key] for r in results], dtype=float)
        ys = np.array([r["si_snr_db"] for r in results])
        ax.scatter(xs, ys, s=10, alpha=0.6)
        if xs.size > 2 and np.ptp(xs) > 0.0:
            slope, intercept = np.polyfit(xs, ys, 1)
            grid = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(grid, slope * grid + intercept, color="red", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_title(label)
    axes[0].set_ylabel("enhanced SI-SNR vs clean (dB)")
    fig.suptitle(f"Clean passthrough SI-SNR vs {xlabel}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def correlation_summary(x, y):
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"pearson_r": None, "spearman_r": None}
    return {
        "pearson_r": float(pearsonr(x, y)[0]),
        "spearman_r": float(spearmanr(x, y)[0]),
    }


def plot_group_attenuation(per_model_results, out_path):
    groups = ["att_voiced_db", "att_low_energy_db", "att_transient_db", "att_steady_db"]
    names = ["voiced", "low-energy", "transient", "steady"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(len(groups))
    for index, (label, results) in enumerate(per_model_results.items()):
        means = [
            np.nanmean([r[g] for r in results if r[g] is not None]) for g in groups
        ]
        ax.bar(x + (index - 1) * width, means, width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean frame attenuation (dB)")
    ax.set_title("Clean passthrough: attenuation by frame group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_worst_spectrograms(worst_files, wavs, models_outputs, fs, n_fft, hop_length, win_length, out_dir, center=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in worst_files:
        input_wav = wavs[name]
        rows = [("input", input_wav)] + [
            (label, models_outputs[label][name]) for label in models_outputs
        ]
        fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.2 * len(rows)), sharex=True)
        specs = []
        for _, wav in rows:
            spec = complex_stft(wav, n_fft, hop_length, win_length, center=center)
            specs.append(20.0 * np.log10(np.abs(spec) + 1e-8))
        vmax = max(float(np.max(s)) for s in specs)
        vmin = vmax - 80.0
        for ax, (label, _), spec_db in zip(axes, rows, specs):
            ax.imshow(
                spec_db,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                extent=[0, len(wav) / fs, 0, fs / 2],
            )
            ax.set_ylabel(label, rotation=0, labelpad=35, va="center")
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(out_dir / f"{Path(name).stem}_spectrograms.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="label=path/to/best.tar; repeatable",
    )
    parser.add_argument("--scene-type", default="clean")
    parser.add_argument("--save-worst", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    with open(args.metadata_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene_type") == args.scene_type:
                meta[row["file"]] = {
                    "speech_activity": float(row["speech_activity"]),
                    "noisy_dbfs": float(row["noisy_dbfs"]),
                    "speaker_id": row.get("speaker_id", ""),
                }
    if not meta:
        raise ValueError(f"No files with scene_type={args.scene_type} in {args.metadata_csv}")
    names = sorted(meta.keys())
    print(f"{len(names)} {args.scene_type} files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy_dir = Path(args.noisy_dir)
    clean_dir = Path(args.clean_dir)

    wavs = {}
    refs = {}
    for name in names:
        wav, file_fs = read_wav(noisy_dir / name, expected_fs=16000)
        ref, _ = read_wav(clean_dir / name, expected_fs=16000)
        wavs[name] = wav
        refs[name] = ref

    per_model_results = {}
    per_model_frames = {}
    models_outputs = {}
    fs = n_fft = hop_length = win_length = center = None
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", {})
        fs = config.get("fs", 16000)
        win_length = config.get("win_length", 160)
        hop_length = config.get("hop_length", 80)
        n_fft = config.get("n_fft", 256)
        center = config.get("center", True)
        model = GTCRN(nfft=n_fft, fs=fs).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        print(f"model {label}: epoch {checkpoint.get('epoch')} from {path}")

        results = []
        frames = []
        outputs = {}
        for index, name in enumerate(names, start=1):
            wav = torch.from_numpy(wavs[name]).to(device)
            with torch.no_grad():
                enhanced = (
                    enhance_waveform(model, wav, n_fft, hop_length, win_length, center=center)
                    .detach()
                    .cpu()
                    .numpy()
                )
            outputs[name] = enhanced
            try:
                result, frame_rec = analyze_file(
                    wavs[name], enhanced, refs[name], fs, n_fft, hop_length, win_length,
                    center=center,
                )
                result["file"] = name
                result["error"] = ""
            except Exception as exc:  # keep going; record the failure
                result = {"file": name, "error": repr(exc)}
                frame_rec = None
            results.append(result)
            if frame_rec is not None:
                frames.append(frame_rec)
            if index % 20 == 0 or index == len(names):
                print(f"  {label}: {index}/{len(names)}")
        per_model_results[label] = results
        per_model_frames[label] = frames
        models_outputs[label] = outputs

    # Per-file metrics table (long format).
    metric_fields = [
        "file", "model", "si_snr_db", "pesq_wb", "stoi",
        "input_si_snr_db", "input_pesq_wb", "input_stoi",
        "si_snr_change_db", "pesq_change", "stoi_change", "overall_gain_db",
        "att_voiced_db", "att_low_energy_db", "att_transient_db", "att_steady_db",
        "band_gain_0_1k", "band_gain_1_4k", "band_gain_4_8k",
        "transfer_mag_0_1k_db", "transfer_mag_1_4k_db", "transfer_mag_4_8k_db",
        "transfer_phase_0_1k_deg", "transfer_phase_1_4k_deg", "transfer_phase_4_8k_deg",
        "speech_activity", "noisy_dbfs", "error",
    ]
    with open(out_dir / "metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for label, results in per_model_results.items():
            for result in results:
                row = {"file": result["file"], "model": label, "error": result["error"]}
                if not result["error"]:
                    row.update(
                        {
                            "si_snr_db": result["si_snr_db"],
                            "pesq_wb": result["pesq_wb"],
                            "stoi": result["stoi"],
                            "input_si_snr_db": result["input_si_snr_db"],
                            "input_pesq_wb": result["input_pesq_wb"],
                            "input_stoi": result["input_stoi"],
                            "si_snr_change_db": result["si_snr_change_db"],
                            "pesq_change": result["pesq_change"],
                            "stoi_change": result["stoi_change"],
                            "overall_gain_db": result["overall_gain_db"],
                            "att_voiced_db": result["att_voiced_db"],
                            "att_low_energy_db": result["att_low_energy_db"],
                            "att_transient_db": result["att_transient_db"],
                            "att_steady_db": result["att_steady_db"],
                            "band_gain_0_1k": result["band_gain_db"]["0-1kHz"],
                            "band_gain_1_4k": result["band_gain_db"]["1-4kHz"],
                            "band_gain_4_8k": result["band_gain_db"]["4-8kHz"],
                            "transfer_mag_0_1k_db": result["transfer_mag_db"]["0-1kHz"],
                            "transfer_mag_1_4k_db": result["transfer_mag_db"]["1-4kHz"],
                            "transfer_mag_4_8k_db": result["transfer_mag_db"]["4-8kHz"],
                            "transfer_phase_0_1k_deg": result["transfer_phase_deg"]["0-1kHz"],
                            "transfer_phase_1_4k_deg": result["transfer_phase_deg"]["1-4kHz"],
                            "transfer_phase_4_8k_deg": result["transfer_phase_deg"]["4-8kHz"],
                            "speech_activity": meta[result["file"]]["speech_activity"],
                            "noisy_dbfs": meta[result["file"]]["noisy_dbfs"],
                        }
                    )
                writer.writerow(row)

    # Pooled analysis.
    analysis = {"files": len(names), "models": {}}
    ok_results = {
        label: [r for r in results if not r["error"]]
        for label, results in per_model_results.items()
    }
    for label, results in ok_results.items():
        si_snr_values = np.array([r["si_snr_db"] for r in results])
        pesq_values = np.array([r["pesq_wb"] for r in results])
        stoi_values = np.array([r["stoi"] for r in results])
        activity = np.array([meta[r["file"]]["speech_activity"] for r in results])
        loudness = np.array([meta[r["file"]]["noisy_dbfs"] for r in results])
        model_summary = {
            "checkpoint": next(p for p in args.checkpoint if p.startswith(label + "=")),
            "mean_si_snr_db": float(np.mean(si_snr_values)),
            "std_si_snr_db": float(np.std(si_snr_values)),
            "min_si_snr_db": float(np.min(si_snr_values)),
            "mean_pesq_wb": float(np.mean(pesq_values)),
            "min_pesq_wb": float(np.min(pesq_values)),
            "mean_stoi": float(np.mean(stoi_values)),
            "mean_input_si_snr_db": float(np.mean([r["input_si_snr_db"] for r in results])),
            "mean_input_pesq_wb": float(np.mean([r["input_pesq_wb"] for r in results])),
            "mean_input_stoi": float(np.mean([r["input_stoi"] for r in results])),
            "mean_si_snr_change_db": float(np.mean([r["si_snr_change_db"] for r in results])),
            "mean_pesq_change": float(np.mean([r["pesq_change"] for r in results])),
            "mean_stoi_change": float(np.mean([r["stoi_change"] for r in results])),
            "mean_overall_gain_db": float(np.mean([r["overall_gain_db"] for r in results])),
            "mean_att_voiced_db": float(np.nanmean([r["att_voiced_db"] for r in results])),
            "mean_att_low_energy_db": float(
                np.nanmean([r["att_low_energy_db"] for r in results if r["att_low_energy_db"] is not None])
            ),
            "mean_att_transient_db": float(
                np.nanmean([r["att_transient_db"] for r in results if r["att_transient_db"] is not None])
            ),
            "mean_att_steady_db": float(
                np.nanmean([r["att_steady_db"] for r in results if r["att_steady_db"] is not None])
            ),
            "mean_band_gain_db": {
                band: float(
                    np.nanmean([r["band_gain_db"][band] for r in results if r["band_gain_db"][band] is not None])
                )
                for band in BAND_NAMES
            },
            "mean_transfer_mag_db": {
                band: float(
                    np.nanmean([r["transfer_mag_db"][band] for r in results if r["transfer_mag_db"][band] is not None])
                )
                for band in BAND_NAMES
            },
            "mean_transfer_phase_deg": {
                band: float(
                    np.nanmean([r["transfer_phase_deg"][band] for r in results if r["transfer_phase_deg"][band] is not None])
                )
                for band in BAND_NAMES
            },
            "corr_si_snr_vs_speech_activity": correlation_summary(
                activity, si_snr_values
            ),
            "corr_si_snr_vs_noisy_dbfs": correlation_summary(
                loudness, si_snr_values
            ),
        }
        analysis["models"][label] = model_summary

    with open(out_dir / "analysis.json", "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    rng = np.random.default_rng(42)
    plot_band_gains(ok_results, plots_dir / "band_gains.png")
    plot_frame_scatter(per_model_frames, plots_dir / "frame_attenuation_vs_energy.png", rng)
    plot_sisnr_hist(ok_results, plots_dir / "sisnr_hist.png")
    plot_correlation(ok_results, meta, "speech_activity", "speech_activity", plots_dir / "corr_speech_activity.png")
    plot_correlation(ok_results, meta, "noisy_dbfs", "input level (dBFS)", plots_dir / "corr_loudness.png")
    plot_group_attenuation(ok_results, plots_dir / "group_attenuation.png")

    # Worst files by the last model's SI-SNR.
    last_label = list(ok_results.keys())[-1]
    worst = sorted(ok_results[last_label], key=lambda r: r["si_snr_db"])[: args.save_worst]
    plot_worst_spectrograms(
        [r["file"] for r in worst], wavs, models_outputs, fs, n_fft, hop_length, win_length,
        out_dir / "worst_spectrograms", center=center,
    )

    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()

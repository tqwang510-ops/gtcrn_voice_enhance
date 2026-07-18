import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audio_utils import read_wav, rms_dbfs, si_snr_db, stft_to_wav, wav_to_stft
from gtcrn import GTCRN
from loss import HybridLoss


DEFAULT_FS = 16000
DEFAULT_WIN_LENGTH = 160
DEFAULT_HOP_LENGTH = 80
DEFAULT_N_FFT = 256


def load_manifest(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    files = data.get("files", data) if isinstance(data, dict) else data
    if not isinstance(files, list) or not files:
        raise ValueError(f"Manifest {path} must contain a non-empty file list")
    return files


def load_scene_files(path, scene_type, invert=False):
    if not path:
        raise ValueError("Scene-aware sampling requires a metadata CSV")
    names = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            matches = row.get("scene_type") == scene_type
            if matches != invert:
                names.append(row["file"])
    if not names:
        relation = "other than" if invert else "equal to"
        raise ValueError(f"No rows with scene_type {relation} {scene_type} in {path}")
    return names


class PairedWavDataset(Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        fs,
        segment_seconds,
        manifest="",
        training=True,
        seed=42,
        min_clean_rms_db=-40.0,
        segment_attempts=10,
        valid_candidates=16,
        max_files=0,
        files=None,
        identity=False,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.fs = fs
        self.segment_samples = int(segment_seconds * fs)
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.min_clean_rms_db = min_clean_rms_db
        self.segment_attempts = segment_attempts
        self.valid_candidates = valid_candidates

        names = list(files) if files is not None else load_manifest(manifest)
        if names is None:
            names = sorted(path.name for path in self.noisy_dir.glob("*.wav"))
        if max_files:
            names = names[:max_files]
        self.noisy_files = [self.noisy_dir / name for name in names]
        self.identity = identity
        if not self.noisy_files:
            raise ValueError(f"No wav files found in {self.noisy_dir}")

        missing = [
            path.name
            for path in self.noisy_files
            if not path.exists() or not (self.clean_dir / path.name).exists()
        ]
        if missing:
            raise ValueError(f"Missing paired wav files: {missing[:5]}")

    def __len__(self):
        return len(self.noisy_files)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _candidate_starts(self, max_start, index):
        if max_start <= 0:
            return [0]
        if self.training:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            return [rng.randint(0, max_start) for _ in range(self.segment_attempts)]
        count = min(self.valid_candidates, max_start + 1)
        return np.linspace(0, max_start, num=count, dtype=np.int64).tolist()

    def _select_start(self, clean, index):
        max_start = len(clean) - self.segment_samples
        candidates = self._candidate_starts(max_start, index)
        best_start = candidates[0]
        best_rms = float("-inf")
        for start in candidates:
            level = rms_dbfs(clean[start : start + self.segment_samples])
            if level > best_rms:
                best_start = start
                best_rms = level
            if self.training and level >= self.min_clean_rms_db:
                return start
        return best_start

    def _crop_or_pad_pair(self, noisy, clean, index):
        length = min(len(noisy), len(clean))
        noisy = noisy[:length]
        clean = clean[:length]

        if length >= self.segment_samples:
            start = self._select_start(clean, index)
            end = start + self.segment_samples
            return noisy[start:end], clean[start:end]

        pad = self.segment_samples - length
        return np.pad(noisy, (0, pad)), np.pad(clean, (0, pad))

    def __getitem__(self, index):
        noisy_path = self.noisy_files[index]
        clean_path = self.clean_dir / noisy_path.name
        noisy, _ = read_wav(noisy_path, self.fs)
        clean, _ = read_wav(clean_path, self.fs)
        noisy, clean = self._crop_or_pad_pair(noisy, clean, index)
        return (
            torch.from_numpy(noisy),
            torch.from_numpy(clean),
            torch.tensor(self.identity, dtype=torch.bool),
        )


class ReplayMixDataset(Dataset):
    def __init__(self, primary, replay, replay_fraction, seed=42, epoch_size=0):
        if not 0.0 < replay_fraction < 1.0:
            raise ValueError("replay_fraction must be between 0 and 1")
        self.primary = primary
        self.replay = replay
        self.replay_fraction = replay_fraction
        self.seed = seed
        self.epoch_size = epoch_size or len(primary)
        if self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        self.schedule = []
        self.set_epoch(0)

    @staticmethod
    def _sample_indices(size, count, rng):
        if count <= size:
            return rng.sample(range(size), count)
        full_repeats, remainder = divmod(count, size)
        indices = list(range(size)) * full_repeats
        indices.extend(rng.sample(range(size), remainder))
        rng.shuffle(indices)
        return indices

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch):
        self.primary.set_epoch(epoch)
        self.replay.set_epoch(epoch)
        rng = random.Random(self.seed + epoch * 1_000_003)
        replay_count = int(round(self.epoch_size * self.replay_fraction))
        primary_count = self.epoch_size - replay_count
        primary_indices = self._sample_indices(len(self.primary), primary_count, rng)
        replay_indices = self._sample_indices(len(self.replay), replay_count, rng)
        self.schedule = [(0, index) for index in primary_indices]
        self.schedule.extend((1, index) for index in replay_indices)
        rng.shuffle(self.schedule)

    def __getitem__(self, index):
        source, source_index = self.schedule[index]
        dataset = self.replay if source else self.primary
        return dataset[source_index]


class SceneAwareMixDataset(Dataset):
    def __init__(
        self,
        primary,
        clean_identity,
        replay,
        clean_fraction,
        replay_fraction,
        seed=42,
        epoch_size=0,
    ):
        if clean_fraction <= 0.0 or replay_fraction <= 0.0:
            raise ValueError("clean and replay fractions must both be positive")
        if clean_fraction + replay_fraction >= 1.0:
            raise ValueError("clean and replay fractions must sum to less than 1")
        self.datasets = [primary, clean_identity, replay]
        self.clean_fraction = clean_fraction
        self.replay_fraction = replay_fraction
        self.seed = seed
        self.epoch_size = epoch_size or len(primary)
        self.schedule = []
        self.set_epoch(0)

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)
        rng = random.Random(self.seed + epoch * 1_000_003)
        clean_count = int(round(self.epoch_size * self.clean_fraction))
        replay_count = int(round(self.epoch_size * self.replay_fraction))
        primary_count = self.epoch_size - clean_count - replay_count
        counts = [primary_count, clean_count, replay_count]
        self.schedule = []
        for source, (dataset, count) in enumerate(zip(self.datasets, counts)):
            indices = ReplayMixDataset._sample_indices(len(dataset), count, rng)
            self.schedule.extend((source, index) for index in indices)
        rng.shuffle(self.schedule)

    def __getitem__(self, index):
        source, source_index = self.schedule[index]
        return self.datasets[source][source_index]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_for_epoch(epoch, args):
    if args.scheduler == "none":
        return args.lr
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        if args.warmup_epochs == 1:
            return args.lr
        progress = (epoch - 1) / (args.warmup_epochs - 1)
        return args.warmup_start_lr + progress * (args.lr - args.warmup_start_lr)
    decay_epochs = max(1, args.epochs - args.warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - args.warmup_epochs) / decay_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def set_learning_rate(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def low_energy_identity_loss(enhanced_stft, clean_stft, identity, args):
    identity = identity.bool()
    if not torch.any(identity):
        return enhanced_stft.new_zeros(())
    enhanced = enhanced_stft[identity]
    clean = clean_stft[identity]
    enhanced_energy = torch.sum(enhanced.square(), dim=(1, 3))
    clean_energy = torch.sum(clean.square(), dim=(1, 3))
    peak = clean_energy.amax(dim=1, keepdim=True).clamp_min(1e-12)
    relative_db = 10.0 * torch.log10((clean_energy / peak).clamp_min(1e-12))
    valid = (
        (relative_db >= args.identity_energy_min_db)
        & (relative_db < args.identity_energy_max_db)
    )
    epsilon = peak * 1e-12
    gain_db = 10.0 * torch.log10(
        ((enhanced_energy + epsilon) / (clean_energy + epsilon)).clamp_min(1e-12)
    )
    gain_db = gain_db.clamp(-args.identity_gain_clamp_db, args.identity_gain_clamp_db)
    weights = valid.to(gain_db.dtype)
    return torch.sum(gain_db.square() * weights) / weights.sum().clamp_min(1.0)


def batch_loss(enhanced_stft, clean_stft, identity, loss_fn, args):
    loss = loss_fn(enhanced_stft, clean_stft)
    identity_loss = low_energy_identity_loss(enhanced_stft, clean_stft, identity, args)
    return loss + args.identity_loss_weight * identity_loss


def run_epoch(model, loader, loss_fn, optimizer, device, args, training):
    model.train(training)
    if training and args.freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    total_items = 0

    for step, (noisy, clean, identity) in enumerate(loader, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        identity = identity.to(device, non_blocking=True)
        noisy_stft = wav_to_stft(
            noisy, args.n_fft, args.hop_length, args.win_length, center=args.center
        )
        clean_stft = wav_to_stft(
            clean, args.n_fft, args.hop_length, args.win_length, center=args.center
        )

        with torch.set_grad_enabled(training):
            enhanced_stft = model(noisy_stft)
            loss = batch_loss(enhanced_stft, clean_stft, identity, loss_fn, args)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_items = noisy.shape[0]
        total_loss += loss.item() * batch_items
        total_items += batch_items

        if training and step % args.log_interval == 0:
            print(f"step {step:05d}/{len(loader):05d} loss={loss.item():.4f}")

    return total_loss / max(1, total_items)


def run_clean_identity_validation(model, loader, loss_fn, device, args):
    from pesq import pesq
    from pystoi import stoi

    model.eval()
    total_loss = 0.0
    total_items = 0
    si_snr_values = []
    pesq_changes = []
    stoi_changes = []
    with torch.no_grad():
        for noisy, clean, identity in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            identity = identity.to(device, non_blocking=True)
            noisy_stft = wav_to_stft(
                noisy, args.n_fft, args.hop_length, args.win_length, center=args.center
            )
            clean_stft = wav_to_stft(
                clean, args.n_fft, args.hop_length, args.win_length, center=args.center
            )
            enhanced_stft = model(noisy_stft)
            loss = batch_loss(enhanced_stft, clean_stft, identity, loss_fn, args)
            enhanced = stft_to_wav(
                enhanced_stft,
                args.n_fft,
                args.hop_length,
                args.win_length,
                length=noisy.shape[-1],
                center=args.center,
            )
            batch_items = noisy.shape[0]
            total_loss += loss.item() * batch_items
            total_items += batch_items
            for item in range(batch_items):
                input_wav = noisy[item].detach().cpu().numpy()
                clean_wav = clean[item].detach().cpu().numpy()
                enhanced_wav = enhanced[item].detach().cpu().numpy()
                si_snr_values.append(si_snr_db(enhanced_wav, clean_wav))
                input_pesq = float(pesq(args.fs, clean_wav, input_wav, "wb"))
                enhanced_pesq = float(pesq(args.fs, clean_wav, enhanced_wav, "wb"))
                pesq_changes.append(enhanced_pesq - input_pesq)
                input_stoi = float(stoi(clean_wav, input_wav, args.fs, extended=False))
                enhanced_stoi = float(
                    stoi(clean_wav, enhanced_wav, args.fs, extended=False)
                )
                stoi_changes.append(enhanced_stoi - input_stoi)
    return {
        "loss": total_loss / max(1, total_items),
        "si_snr_db": float(np.mean(si_snr_values)),
        "pesq_change": float(np.mean(pesq_changes)),
        "stoi_change": float(np.mean(stoi_changes)),
    }


def clean_gate_passed(metrics, args):
    minimum_si_snr = args.clean_gate_min_si_snr
    if args.clean_gate_reference_si_snr > 0.0:
        minimum_si_snr = max(
            minimum_si_snr,
            args.clean_gate_reference_si_snr - args.clean_gate_max_si_snr_drop,
        )
    return (
        metrics["si_snr_db"] >= minimum_si_snr
        and metrics["pesq_change"] >= args.clean_gate_min_pesq_change
        and metrics["stoi_change"] >= args.clean_gate_min_stoi_change
    )


def load_validation_domains(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    domains = config.get("domains") if isinstance(config, dict) else config
    if not domains:
        raise ValueError(f"Validation domains file {path} has no domains")
    names = [domain["name"] for domain in domains]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate validation domain names: {names}")
    return domains


def domain_scene_files(
    metadata_csv,
    include_scene_types,
    exclude_scene_types,
    require_nonempty_fields=None,
    require_empty_fields=None,
    numeric_filters=None,
):
    require_nonempty_fields = require_nonempty_fields or []
    require_empty_fields = require_empty_fields or []
    numeric_filters = numeric_filters or {}
    names = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scene = row.get("scene_type", "")
            if include_scene_types and scene not in include_scene_types:
                continue
            if exclude_scene_types and scene in exclude_scene_types:
                continue
            if any(not row.get(field, "").strip() for field in require_nonempty_fields):
                continue
            if any(row.get(field, "").strip() for field in require_empty_fields):
                continue
            numeric_match = True
            for field, limits in numeric_filters.items():
                raw_value = row.get(field, "").strip()
                if not raw_value:
                    numeric_match = False
                    break
                value = float(raw_value)
                if limits.get("min") is not None and value < float(limits["min"]):
                    numeric_match = False
                    break
                if limits.get("max") is not None and value > float(limits["max"]):
                    numeric_match = False
                    break
            if not numeric_match:
                continue
            names.append(row["file"])
    if not names:
        raise ValueError(f"No files left after scene filtering in {metadata_csv}")
    return names


def build_domain_dataset(domain, args):
    files = None
    if domain.get("metadata_csv"):
        files = domain_scene_files(
            domain["metadata_csv"],
            domain.get("include_scene_types") or [],
            domain.get("exclude_scene_types") or [],
            domain.get("require_nonempty_fields") or [],
            domain.get("require_empty_fields") or [],
            domain.get("numeric_filters") or {},
        )
    elif domain.get("manifest"):
        files = load_manifest(domain["manifest"])
    max_files = int(domain.get("max_files", 0))
    if files is not None and max_files and len(files) > max_files:
        files = list(files)
        sample_seed = int(domain.get("sample_seed", args.seed))
        random.Random(sample_seed).shuffle(files)
        files = files[:max_files]
    return PairedWavDataset(
        domain["noisy"],
        domain["clean"],
        args.fs,
        args.segment_seconds,
        manifest="" if files is not None else domain.get("manifest", ""),
        training=False,
        seed=args.seed,
        min_clean_rms_db=args.min_clean_rms_db,
        segment_attempts=args.segment_attempts,
        valid_candidates=args.valid_candidates,
        max_files=0 if files is not None else max_files,
        files=files,
        identity=bool(domain.get("identity", False)),
    )


def run_domain_validation(model, loader, loss_fn, device, args, identity):
    from pesq import NoUtterancesError, pesq
    from pystoi import stoi

    model.eval()
    total_loss = 0.0
    total_items = 0
    si_snr_values = []
    si_snr_changes = []
    pesq_changes = []
    stoi_changes = []
    pesq_skipped_no_utterances = 0
    with torch.no_grad():
        for noisy, clean, identity_flag in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            identity_flag = identity_flag.to(device, non_blocking=True)
            noisy_stft = wav_to_stft(
                noisy, args.n_fft, args.hop_length, args.win_length, center=args.center
            )
            clean_stft = wav_to_stft(
                clean, args.n_fft, args.hop_length, args.win_length, center=args.center
            )
            enhanced_stft = model(noisy_stft)
            loss = batch_loss(enhanced_stft, clean_stft, identity_flag, loss_fn, args)
            enhanced = stft_to_wav(
                enhanced_stft,
                args.n_fft,
                args.hop_length,
                args.win_length,
                length=noisy.shape[-1],
                center=args.center,
            )
            batch_items = noisy.shape[0]
            total_loss += loss.item() * batch_items
            total_items += batch_items
            for item in range(batch_items):
                input_wav = noisy[item].detach().cpu().numpy()
                clean_wav = clean[item].detach().cpu().numpy()
                enhanced_wav = enhanced[item].detach().cpu().numpy()
                si_snr_values.append(si_snr_db(enhanced_wav, clean_wav))
                if not identity:
                    si_snr_changes.append(
                        si_snr_db(enhanced_wav, clean_wav)
                        - si_snr_db(input_wav, clean_wav)
                    )
                try:
                    input_pesq = float(pesq(args.fs, clean_wav, input_wav, "wb"))
                    enhanced_pesq = float(pesq(args.fs, clean_wav, enhanced_wav, "wb"))
                    pesq_changes.append(enhanced_pesq - input_pesq)
                except NoUtterancesError:
                    pesq_skipped_no_utterances += 1
                input_stoi = float(stoi(clean_wav, input_wav, args.fs, extended=False))
                enhanced_stoi = float(
                    stoi(clean_wav, enhanced_wav, args.fs, extended=False)
                )
                if not (math.isfinite(input_stoi) and math.isfinite(enhanced_stoi)):
                    raise ValueError("STOI returned a non-finite value during domain validation")
                stoi_changes.append(enhanced_stoi - input_stoi)
    metrics = {
        "loss": total_loss / max(1, total_items),
        "si_snr_db": float(np.mean(si_snr_values)),
        "pesq_change": float(np.mean(pesq_changes)) if pesq_changes else float("nan"),
        "stoi_change": float(np.mean(stoi_changes)) if stoi_changes else float("nan"),
        "items": total_items,
        "pesq_items": len(pesq_changes),
        "pesq_skipped_no_utterances": pesq_skipped_no_utterances,
        "stoi_items": len(stoi_changes),
    }
    if si_snr_values:
        metrics["si_snr_median_db"] = float(np.median(si_snr_values))
        metrics["si_snr_p10_db"] = float(np.percentile(si_snr_values, 10))
        metrics["si_snr_below_20_db_fraction"] = float(
            np.mean(np.asarray(si_snr_values) < 20.0)
        )
        metrics["si_snr_below_30_db_fraction"] = float(
            np.mean(np.asarray(si_snr_values) < 30.0)
        )
    if not identity:
        metrics["si_snr_change"] = float(np.mean(si_snr_changes))
    return metrics


def domain_gate_passed(metrics, gate):
    lower_bounds = [
        ("min_si_snr_db", "si_snr_db"),
        ("min_si_snr_change", "si_snr_change"),
        ("min_pesq_change", "pesq_change"),
        ("min_stoi_change", "stoi_change"),
        ("min_si_snr_p10_db", "si_snr_p10_db"),
    ]
    for gate_key, metric_key in lower_bounds:
        threshold = gate.get(gate_key)
        if threshold is None:
            continue
        value = metrics.get(metric_key)
        if value is None or not math.isfinite(value) or value < threshold:
            return False
    threshold = gate.get("max_loss")
    if threshold is not None:
        value = metrics.get("loss")
        if value is None or not math.isfinite(value) or value > threshold:
            return False
    return True


def domain_selection_value(metrics, domain):
    selection = domain.get("selection")
    if not selection:
        return metrics["loss"]
    metric_name = selection["metric"]
    value = metrics.get(metric_name)
    if value is None or not math.isfinite(value):
        raise ValueError(
            f"Selection metric {metric_name!r} is unavailable for domain {domain['name']!r}"
        )
    scale = float(selection.get("scale", 1.0))
    if scale <= 0.0:
        raise ValueError(f"Selection scale must be positive for domain {domain['name']!r}")
    reference = float(selection.get("reference", 0.0))
    normalized = (value - reference) / scale
    if selection.get("direction", "max") == "max":
        return -normalized
    if selection["direction"] == "min":
        return normalized
    raise ValueError(f"Unknown selection direction for domain {domain['name']!r}")


def checkpoint_config(args):
    return {
        "fs": args.fs,
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "center": args.center,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    args,
    valid_loss,
    best_loss,
    validation_metrics=None,
    epochs_without_improvement=0,
    best_selection_loss=float("inf"),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
            "best_loss": best_loss,
            "best_selection_loss": best_selection_loss,
            "validation_metrics": validation_metrics or {"valid_loss": valid_loss},
            "epochs_without_improvement": epochs_without_improvement,
            "config": checkpoint_config(args),
            "training_config": vars(args),
        },
        path,
    )


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def load_history(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def save_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    fieldnames = ["epoch", "train_loss"]
    trailing = ["valid_loss", "selection_loss", "learning_rate", "seconds"]
    middle = sorted(
        key
        for key in {key for row in history for key in row}
        if key not in fieldnames + trailing
    )
    fieldnames.extend(middle)
    fieldnames.extend(
        field for field in trailing if any(field in row for row in history)
    )
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def plot_history(path, history):
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    valid_loss = [float(row["valid_loss"]) for row in history]
    learning_rate = [float(row["learning_rate"]) for row in history]

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(epochs, train_loss, marker="o", markersize=3, label="train loss")
    axes[0].plot(epochs, valid_loss, marker="o", markersize=3, label="selection loss")
    if "primary_valid_loss" in history[-1]:
        primary_valid_loss = [float(row["primary_valid_loss"]) for row in history]
        replay_valid_loss = [float(row["replay_valid_loss"]) for row in history]
        axes[0].plot(
            epochs,
            primary_valid_loss,
            marker="o",
            markersize=3,
            label="classroom valid loss",
        )
        axes[0].plot(
            epochs,
            replay_valid_loss,
            marker="o",
            markersize=3,
            label="replay valid loss",
        )
    axes[0].set_ylabel("HybridLoss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, learning_rate, marker="o", markersize=3, color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_clean_history(path, history, args):
    if not history or "clean_si_snr_db" not in history[-1]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    si_snr = [float(row["clean_si_snr_db"]) for row in history]
    pesq_change = [float(row["clean_pesq_change"]) for row in history]
    stoi_change = [float(row["clean_stoi_change"]) for row in history]
    minimum_si_snr = args.clean_gate_min_si_snr
    if args.clean_gate_reference_si_snr > 0.0:
        minimum_si_snr = max(
            minimum_si_snr,
            args.clean_gate_reference_si_snr - args.clean_gate_max_si_snr_drop,
        )
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    series = [
        (si_snr, minimum_si_snr, "clean SI-SNR (dB)"),
        (pesq_change, args.clean_gate_min_pesq_change, "clean PESQ change"),
        (stoi_change, args.clean_gate_min_stoi_change, "clean STOI change"),
    ]
    for axis, (values, threshold, label) in zip(axes, series):
        axis.plot(epochs, values, marker="o", markersize=4)
        axis.axhline(threshold, color="tab:red", linestyle="--", label="hard gate")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Train GTCRN on paired noisy/clean wav files.")
    parser.add_argument("--train-noisy", required=True)
    parser.add_argument("--train-clean", required=True)
    parser.add_argument("--valid-noisy", required=True)
    parser.add_argument("--valid-clean", required=True)
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--valid-manifest", default="")
    parser.add_argument("--train-metadata-csv", default="")
    parser.add_argument("--valid-metadata-csv", default="")
    parser.add_argument("--replay-train-noisy", default="")
    parser.add_argument("--replay-train-clean", default="")
    parser.add_argument("--replay-train-manifest", default="")
    parser.add_argument("--replay-valid-noisy", default="")
    parser.add_argument("--replay-valid-clean", default="")
    parser.add_argument("--replay-valid-manifest", default="")
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    parser.add_argument("--clean-fraction", type=float, default=0.0)
    parser.add_argument("--epoch-size", type=int, default=0)
    parser.add_argument("--max-train-files", type=int, default=0)
    parser.add_argument("--max-valid-files", type=int, default=0)
    parser.add_argument("--max-replay-train-files", type=int, default=0)
    parser.add_argument("--max-replay-valid-files", type=int, default=0)
    parser.add_argument("--max-clean-train-files", type=int, default=0)
    parser.add_argument("--max-clean-valid-files", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/voicebank_serious")
    parser.add_argument("--fs", type=int, default=DEFAULT_FS)
    parser.add_argument("--win-length", type=int, default=DEFAULT_WIN_LENGTH)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--min-clean-rms-db", type=float, default=-40.0)
    parser.add_argument("--segment-attempts", type=int, default=10)
    parser.add_argument("--valid-candidates", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--identity-loss-weight", type=float, default=0.0)
    parser.add_argument("--identity-energy-min-db", type=float, default=-50.0)
    parser.add_argument("--identity-energy-max-db", type=float, default=-20.0)
    parser.add_argument("--identity-gain-clamp-db", type=float, default=20.0)
    parser.add_argument("--primary-valid-weight", type=float, default=0.60)
    parser.add_argument("--replay-valid-weight", type=float, default=0.25)
    parser.add_argument("--clean-valid-weight", type=float, default=0.15)
    parser.add_argument("--clean-gate-min-si-snr", type=float, default=65.0)
    parser.add_argument("--clean-gate-min-pesq-change", type=float, default=-0.03)
    parser.add_argument("--clean-gate-min-stoi-change", type=float, default=-0.001)
    parser.add_argument("--clean-gate-reference-si-snr", type=float, default=0.0)
    parser.add_argument("--clean-gate-max-si-snr-drop", type=float, default=10.0)
    parser.add_argument(
        "--validation-domains",
        default="",
        help=(
            "JSON file listing extra validation domains with per-domain hard "
            "gates. When set, it replaces the built-in primary/replay/clean "
            "validation selection logic."
        ),
    )
    parser.add_argument(
        "--clean-scene-type",
        default="clean",
        help="scene_type value treated as clean/identity in metadata CSVs",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both.")
    replay_train_paths = [args.replay_train_noisy, args.replay_train_clean]
    replay_valid_paths = [args.replay_valid_noisy, args.replay_valid_clean]
    replay_enabled = any(replay_train_paths + replay_valid_paths) or args.replay_fraction > 0
    scene_aware = args.clean_fraction > 0.0
    if replay_enabled:
        if not all(replay_train_paths + replay_valid_paths):
            raise ValueError(
                "Replay requires train/valid noisy and clean directories."
            )
        if not 0.0 < args.replay_fraction < 1.0:
            raise ValueError("--replay-fraction must be between 0 and 1")
    if scene_aware:
        if not replay_enabled:
            raise ValueError("Scene-aware repair requires VoiceBank replay")
        if not args.train_metadata_csv or not args.valid_metadata_csv:
            raise ValueError(
                "--clean-fraction requires --train-metadata-csv and --valid-metadata-csv"
            )
        if args.clean_fraction + args.replay_fraction >= 1.0:
            raise ValueError("clean and replay fractions must sum to less than 1")
        valid_weight_sum = (
            args.primary_valid_weight
            + args.replay_valid_weight
            + args.clean_valid_weight
        )
        if not math.isclose(valid_weight_sum, 1.0, abs_tol=1e-6):
            raise ValueError("validation weights must sum to 1")
        if args.identity_loss_weight <= 0.0:
            raise ValueError("Scene-aware repair requires --identity-loss-weight > 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists() and not args.resume and not args.overwrite_run:
        raise FileExistsError(
            f"{metrics_path} already exists. Use a new --out-dir, --resume, or --overwrite-run."
        )

    print(f"device={device}")
    print(
        f"STFT fs={args.fs}, win_length={args.win_length}, hop_length={args.hop_length}, "
        f"n_fft={args.n_fft}, center={args.center}"
    )

    non_clean_train_files = None
    clean_train_files = None
    non_clean_valid_files = None
    clean_valid_files = None
    if scene_aware:
        non_clean_train_files = load_scene_files(
            args.train_metadata_csv, args.clean_scene_type, invert=True
        )
        clean_train_files = load_scene_files(args.train_metadata_csv, args.clean_scene_type)
        non_clean_valid_files = load_scene_files(
            args.valid_metadata_csv, args.clean_scene_type, invert=True
        )
        clean_valid_files = load_scene_files(args.valid_metadata_csv, args.clean_scene_type)

    primary_train_set = PairedWavDataset(
        args.train_noisy,
        args.train_clean,
        args.fs,
        args.segment_seconds,
        manifest=args.train_manifest,
        training=True,
        seed=args.seed,
        min_clean_rms_db=args.min_clean_rms_db,
        segment_attempts=args.segment_attempts,
        valid_candidates=args.valid_candidates,
        max_files=args.max_train_files,
        files=non_clean_train_files,
    )
    valid_set = PairedWavDataset(
        args.valid_noisy,
        args.valid_clean,
        args.fs,
        args.segment_seconds,
        manifest=args.valid_manifest,
        training=False,
        seed=args.seed,
        min_clean_rms_db=args.min_clean_rms_db,
        segment_attempts=args.segment_attempts,
        valid_candidates=args.valid_candidates,
        max_files=args.max_valid_files,
        files=non_clean_valid_files,
    )
    clean_train_set = None
    clean_valid_set = None
    if scene_aware:
        clean_train_set = PairedWavDataset(
            args.train_noisy,
            args.train_clean,
            args.fs,
            args.segment_seconds,
            training=True,
            seed=args.seed + 31,
            min_clean_rms_db=args.min_clean_rms_db,
            segment_attempts=args.segment_attempts,
            valid_candidates=args.valid_candidates,
            files=clean_train_files,
            identity=True,
            max_files=args.max_clean_train_files,
        )
        clean_valid_set = PairedWavDataset(
            args.valid_noisy,
            args.valid_clean,
            args.fs,
            args.segment_seconds,
            training=False,
            seed=args.seed + 31,
            min_clean_rms_db=args.min_clean_rms_db,
            segment_attempts=args.segment_attempts,
            valid_candidates=args.valid_candidates,
            files=clean_valid_files,
            identity=True,
            max_files=args.max_clean_valid_files,
        )
    replay_train_set = None
    replay_valid_set = None
    if replay_enabled:
        replay_train_set = PairedWavDataset(
            args.replay_train_noisy,
            args.replay_train_clean,
            args.fs,
            args.segment_seconds,
            manifest=args.replay_train_manifest,
            training=True,
            seed=args.seed + 17,
            min_clean_rms_db=args.min_clean_rms_db,
            segment_attempts=args.segment_attempts,
            valid_candidates=args.valid_candidates,
            max_files=args.max_replay_train_files,
        )
        replay_valid_set = PairedWavDataset(
            args.replay_valid_noisy,
            args.replay_valid_clean,
            args.fs,
            args.segment_seconds,
            manifest=args.replay_valid_manifest,
            training=False,
            seed=args.seed + 17,
            min_clean_rms_db=args.min_clean_rms_db,
            segment_attempts=args.segment_attempts,
            valid_candidates=args.valid_candidates,
            max_files=args.max_replay_valid_files,
        )
        if scene_aware:
            train_set = SceneAwareMixDataset(
                primary_train_set,
                clean_train_set,
                replay_train_set,
                args.clean_fraction,
                args.replay_fraction,
                seed=args.seed,
                epoch_size=args.epoch_size,
            )
        else:
            train_set = ReplayMixDataset(
                primary_train_set,
                replay_train_set,
                args.replay_fraction,
                seed=args.seed,
                epoch_size=args.epoch_size,
            )
    else:
        train_set = primary_train_set
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=not replay_enabled,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    replay_valid_loader = None
    if replay_valid_set is not None:
        replay_valid_loader = DataLoader(
            replay_valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
    clean_valid_loader = None
    if clean_valid_set is not None:
        clean_valid_loader = DataLoader(
            clean_valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
    domains = None
    domain_loaders = None
    if args.validation_domains:
        domains = load_validation_domains(args.validation_domains)
        domain_loaders = [
            DataLoader(
                build_domain_dataset(domain, args),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=False,
            )
            for domain in domains
        ]

    model = GTCRN(nfft=args.n_fft, fs=args.fs).to(device)
    loss_fn = HybridLoss(
        args.n_fft, args.hop_length, args.win_length, center=args.center
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    best_loss = float("inf")
    best_selection_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        saved_config = checkpoint.get("config", {})
        requested_config = checkpoint_config(args)
        mismatches = {
            key: (saved_config[key], value)
            for key, value in requested_config.items()
            if key in saved_config and saved_config[key] != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint STFT configuration mismatch: {mismatches}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("valid_loss", best_loss)))
        best_selection_loss = float(
            checkpoint.get("best_selection_loss", best_selection_loss)
        )
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history = [
            row for row in load_history(metrics_path) if int(row["epoch"]) < start_epoch
        ]
        print(f"resumed from {args.resume} after epoch {start_epoch - 1:03d}")
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        saved_config = checkpoint.get("config", {})
        requested_config = checkpoint_config(args)
        mismatches = {
            key: (saved_config[key], value)
            for key, value in requested_config.items()
            if key in saved_config and saved_config[key] != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint STFT configuration mismatch: {mismatches}")
        model.load_state_dict(checkpoint["model"])
        print(
            f"initialized model from {args.init_checkpoint}; "
            "optimizer and epoch start from scratch"
        )
    elif args.overwrite_run:
        history = []

    run_config = vars(args).copy()
    run_config["command"] = " ".join(sys.argv)
    run_config["device"] = str(device)
    run_config["train_files"] = len(train_set)
    run_config["primary_train_files"] = len(primary_train_set)
    run_config["valid_files"] = len(valid_set)
    if replay_enabled:
        run_config["replay_train_files"] = len(replay_train_set)
        run_config["replay_valid_files"] = len(replay_valid_set)
        run_config["replay_items_per_epoch"] = int(
            round(len(train_set) * args.replay_fraction)
        )
        if not scene_aware:
            run_config["primary_items_per_epoch"] = (
                len(train_set) - run_config["replay_items_per_epoch"]
            )
    if scene_aware:
        run_config["clean_train_files"] = len(clean_train_set)
        run_config["clean_valid_files"] = len(clean_valid_set)
        run_config["clean_items_per_epoch"] = int(
            round(len(train_set) * args.clean_fraction)
        )
        run_config["non_clean_items_per_epoch"] = (
            len(train_set)
            - run_config["replay_items_per_epoch"]
            - run_config["clean_items_per_epoch"]
        )
        run_config["primary_items_per_epoch"] = run_config[
            "non_clean_items_per_epoch"
        ]
    if domains is not None:
        run_config["validation_domain_names"] = [domain["name"] for domain in domains]
        run_config["validation_domain_files"] = {
            domain["name"]: len(loader.dataset)
            for domain, loader in zip(domains, domain_loaders)
        }
    save_json(out_dir / "config.json", run_config)

    if start_epoch > args.epochs:
        print(f"checkpoint already reached epoch {start_epoch - 1:03d}; nothing to train")
        return

    checkpoints_dir = out_dir / "checkpoints"
    for epoch in range(start_epoch, args.epochs + 1):
        train_set.set_epoch(epoch)
        learning_rate = learning_rate_for_epoch(epoch, args)
        set_learning_rate(optimizer, learning_rate)
        started = time.perf_counter()

        train_loss = run_epoch(
            model, train_loader, loss_fn, optimizer, device, args, training=True
        )
        with torch.no_grad():
            if domain_loaders is not None:
                domain_metrics = {}
                selection_loss = 0.0
                selection_weight = 0.0
                gate_passed = True
                for domain, loader in zip(domains, domain_loaders):
                    metrics = run_domain_validation(
                        model,
                        loader,
                        loss_fn,
                        device,
                        args,
                        bool(domain.get("identity", False)),
                    )
                    domain_metrics[domain["name"]] = metrics
                    weight = float(domain.get("weight", 1.0))
                    selection_loss += weight * domain_selection_value(metrics, domain)
                    selection_weight += weight
                    gate_passed = gate_passed and domain_gate_passed(
                        metrics, domain.get("gate", {})
                    )
                selection_loss /= max(selection_weight, 1e-12)
                primary_valid_loss = selection_loss
                replay_valid_loss = None
                clean_metrics = None
                clean_valid_loss = None
            else:
                domain_metrics = None
                primary_valid_loss = run_epoch(
                    model, valid_loader, loss_fn, optimizer, device, args, training=False
                )
                if replay_valid_loader is not None:
                    replay_valid_loss = run_epoch(
                        model,
                        replay_valid_loader,
                        loss_fn,
                        optimizer,
                        device,
                        args,
                        training=False,
                    )
                    if clean_valid_loader is not None:
                        clean_metrics = run_clean_identity_validation(
                            model, clean_valid_loader, loss_fn, device, args
                        )
                        clean_valid_loss = clean_metrics["loss"]
                        selection_loss = (
                            args.primary_valid_weight * primary_valid_loss
                            + args.replay_valid_weight * replay_valid_loss
                            + args.clean_valid_weight * clean_valid_loss
                        )
                        gate_passed = clean_gate_passed(clean_metrics, args)
                    else:
                        clean_metrics = None
                        clean_valid_loss = None
                        gate_passed = True
                        selection_loss = (
                            (1.0 - args.replay_fraction) * primary_valid_loss
                            + args.replay_fraction * replay_valid_loss
                        )
                else:
                    replay_valid_loss = None
                    clean_metrics = None
                    clean_valid_loss = None
                    gate_passed = True
                    selection_loss = primary_valid_loss
        valid_loss = selection_loss
        seconds = time.perf_counter() - started

        is_best_selection = valid_loss < best_selection_loss
        if is_best_selection:
            best_selection_loss = valid_loss
        is_best = gate_passed and valid_loss < best_loss
        if is_best:
            best_loss = valid_loss
            epochs_without_improvement = 0
        elif math.isfinite(best_loss):
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.8f}",
            "valid_loss": f"{valid_loss:.8f}",
            "learning_rate": f"{learning_rate:.10g}",
            "seconds": f"{seconds:.3f}",
        }
        if replay_valid_loss is not None:
            row.update(
                {
                    "primary_valid_loss": f"{primary_valid_loss:.8f}",
                    "replay_valid_loss": f"{replay_valid_loss:.8f}",
                    "selection_loss": f"{selection_loss:.8f}",
                }
            )
        if clean_metrics is not None:
            row.update(
                {
                    "clean_valid_loss": f"{clean_valid_loss:.8f}",
                    "clean_si_snr_db": f"{clean_metrics['si_snr_db']:.8f}",
                    "clean_pesq_change": f"{clean_metrics['pesq_change']:.8f}",
                    "clean_stoi_change": f"{clean_metrics['stoi_change']:.8f}",
                    "clean_gate_passed": str(bool(gate_passed)).lower(),
                }
            )
        if domain_metrics is not None:
            for domain in domains:
                name = domain["name"]
                metrics = domain_metrics[name]
                row[f"{name}_loss"] = f"{metrics['loss']:.8f}"
                row[f"{name}_si_snr_db"] = f"{metrics['si_snr_db']:.8f}"
                if "si_snr_change" in metrics:
                    row[f"{name}_si_snr_change"] = f"{metrics['si_snr_change']:.8f}"
                row[f"{name}_pesq_change"] = f"{metrics['pesq_change']:.8f}"
                row[f"{name}_stoi_change"] = f"{metrics['stoi_change']:.8f}"
                row[f"{name}_items"] = str(metrics["items"])
                row[f"{name}_pesq_items"] = str(metrics["pesq_items"])
                row[f"{name}_pesq_skipped"] = str(
                    metrics["pesq_skipped_no_utterances"]
                )
                row[f"{name}_stoi_items"] = str(metrics["stoi_items"])
                row[f"{name}_si_snr_p10_db"] = f"{metrics['si_snr_p10_db']:.8f}"
                row[f"{name}_gate_passed"] = str(
                    bool(domain_gate_passed(metrics, domain.get("gate", {})))
                ).lower()
        validation_metrics = {
            "primary_valid_loss": primary_valid_loss,
            "selection_loss": selection_loss,
        }
        if replay_valid_loss is not None:
            validation_metrics["replay_valid_loss"] = replay_valid_loss
        if clean_metrics is not None:
            validation_metrics["clean_identity"] = clean_metrics
            validation_metrics["clean_gate_passed"] = gate_passed
        if domain_metrics is not None:
            validation_metrics["domains"] = domain_metrics
            validation_metrics["domain_gate_passed"] = gate_passed
        history.append(row)
        save_history(metrics_path, history)
        plot_history(out_dir / "training_curve.png", history)
        plot_clean_history(out_dir / "clean_validation_curve.png", history, args)
        save_checkpoint(
            checkpoints_dir / "last.tar",
            model,
            optimizer,
            epoch,
            args,
            valid_loss,
            best_loss,
            validation_metrics,
            epochs_without_improvement,
            best_selection_loss,
        )
        if args.save_every_epoch:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:03d}.tar",
                model,
                optimizer,
                epoch,
                args,
                valid_loss,
                best_loss,
                validation_metrics,
                epochs_without_improvement,
                best_selection_loss,
            )
        if is_best_selection:
            save_checkpoint(
                checkpoints_dir / "best_selection_candidate.tar",
                model,
                optimizer,
                epoch,
                args,
                valid_loss,
                best_loss,
                validation_metrics,
                epochs_without_improvement,
                best_selection_loss,
            )
        if is_best:
            save_checkpoint(
                checkpoints_dir / "best.tar",
                model,
                optimizer,
                epoch,
                args,
                valid_loss,
                best_loss,
                validation_metrics,
                epochs_without_improvement,
                best_selection_loss,
            )

        if domain_metrics is not None:
            domain_text = " ".join(
                f"{name}={domain_metrics[name]['loss']:.4f}" for name in domain_metrics
            )
            validation_text = (
                f"{domain_text} selection_loss={selection_loss:.4f} "
                f"domain_gate={'pass' if gate_passed else 'fail'}"
            )
        elif replay_valid_loss is None:
            validation_text = f"valid_loss={valid_loss:.4f}"
        else:
            validation_text = (
                f"classroom_valid={primary_valid_loss:.4f} "
                f"replay_valid={replay_valid_loss:.4f} "
                f"selection_loss={selection_loss:.4f}"
            )
            if clean_metrics is not None:
                validation_text += (
                    f" clean_valid={clean_valid_loss:.4f} "
                    f"clean_si_snr={clean_metrics['si_snr_db']:.2f}dB "
                    f"clean_pesq_change={clean_metrics['pesq_change']:+.4f} "
                    f"clean_stoi_change={clean_metrics['stoi_change']:+.5f} "
                    f"clean_gate={'pass' if gate_passed else 'fail'}"
                )
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} {validation_text} "
            f"lr={learning_rate:.3e} seconds={seconds:.1f}"
        )
        if is_best:
            print(f"saved best checkpoint: valid_loss={best_loss:.4f}")
        elif is_best_selection:
            print(
                "saved best_selection_candidate.tar; clean hard gate still failed"
            )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping after {epochs_without_improvement} epochs "
                "without validation improvement"
            )
            break


if __name__ == "__main__":
    main()

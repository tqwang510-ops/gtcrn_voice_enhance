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

from audio_utils import read_wav, rms_dbfs, wav_to_stft
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

        names = load_manifest(manifest)
        if names is None:
            names = sorted(path.name for path in self.noisy_dir.glob("*.wav"))
        if max_files:
            names = names[:max_files]
        self.noisy_files = [self.noisy_dir / name for name in names]
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
        return torch.from_numpy(noisy), torch.from_numpy(clean)


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


def run_epoch(model, loader, loss_fn, optimizer, device, args, training):
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for step, (noisy, clean) in enumerate(loader, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        noisy_stft = wav_to_stft(
            noisy, args.n_fft, args.hop_length, args.win_length, center=args.center
        )
        clean_stft = wav_to_stft(
            clean, args.n_fft, args.hop_length, args.win_length, center=args.center
        )

        with torch.set_grad_enabled(training):
            enhanced_stft = model(noisy_stft)
            loss = loss_fn(enhanced_stft, clean_stft)

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
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
            "best_loss": best_loss,
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
    optional_fields = ["primary_valid_loss", "replay_valid_loss", "selection_loss"]
    fieldnames.extend(field for field in optional_fields if field in history[-1])
    fieldnames.extend(["valid_loss", "learning_rate", "seconds"])
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


def main():
    parser = argparse.ArgumentParser(description="Train GTCRN on paired noisy/clean wav files.")
    parser.add_argument("--train-noisy", required=True)
    parser.add_argument("--train-clean", required=True)
    parser.add_argument("--valid-noisy", required=True)
    parser.add_argument("--valid-clean", required=True)
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--valid-manifest", default="")
    parser.add_argument("--replay-train-noisy", default="")
    parser.add_argument("--replay-train-clean", default="")
    parser.add_argument("--replay-train-manifest", default="")
    parser.add_argument("--replay-valid-noisy", default="")
    parser.add_argument("--replay-valid-clean", default="")
    parser.add_argument("--replay-valid-manifest", default="")
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    parser.add_argument("--epoch-size", type=int, default=0)
    parser.add_argument("--max-train-files", type=int, default=0)
    parser.add_argument("--max-valid-files", type=int, default=0)
    parser.add_argument("--max-replay-train-files", type=int, default=0)
    parser.add_argument("--max-replay-valid-files", type=int, default=0)
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
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both.")
    replay_train_paths = [args.replay_train_noisy, args.replay_train_clean]
    replay_valid_paths = [args.replay_valid_noisy, args.replay_valid_clean]
    replay_enabled = any(replay_train_paths + replay_valid_paths) or args.replay_fraction > 0
    if replay_enabled:
        if not all(replay_train_paths + replay_valid_paths):
            raise ValueError(
                "Replay requires train/valid noisy and clean directories."
            )
        if not 0.0 < args.replay_fraction < 1.0:
            raise ValueError("--replay-fraction must be between 0 and 1")
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

    model = GTCRN(nfft=args.n_fft, fs=args.fs).to(device)
    loss_fn = HybridLoss(
        args.n_fft, args.hop_length, args.win_length, center=args.center
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 1
    best_loss = float("inf")
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
        run_config["primary_items_per_epoch"] = len(train_set) - int(
            round(len(train_set) * args.replay_fraction)
        )
        run_config["replay_items_per_epoch"] = int(
            round(len(train_set) * args.replay_fraction)
        )
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
                selection_loss = (
                    (1.0 - args.replay_fraction) * primary_valid_loss
                    + args.replay_fraction * replay_valid_loss
                )
            else:
                replay_valid_loss = None
                selection_loss = primary_valid_loss
        valid_loss = selection_loss
        seconds = time.perf_counter() - started

        is_best = valid_loss < best_loss
        if is_best:
            best_loss = valid_loss
            epochs_without_improvement = 0
        else:
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
        validation_metrics = {
            "primary_valid_loss": primary_valid_loss,
            "selection_loss": selection_loss,
        }
        if replay_valid_loss is not None:
            validation_metrics["replay_valid_loss"] = replay_valid_loss
        history.append(row)
        save_history(metrics_path, history)
        plot_history(out_dir / "training_curve.png", history)
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
            )

        if replay_valid_loss is None:
            validation_text = f"valid_loss={valid_loss:.4f}"
        else:
            validation_text = (
                f"classroom_valid={primary_valid_loss:.4f} "
                f"replay_valid={replay_valid_loss:.4f} "
                f"selection_loss={selection_loss:.4f}"
            )
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} {validation_text} "
            f"lr={learning_rate:.3e} seconds={seconds:.1f}"
        )
        if is_best:
            print(f"saved best checkpoint: valid_loss={best_loss:.4f}")
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

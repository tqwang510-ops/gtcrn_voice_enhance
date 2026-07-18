import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gtcrn import GTCRN
from loss import HybridLoss
from train_custom import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_WIN_LENGTH,
    build_domain_dataset,
    domain_gate_passed,
    domain_selection_value,
    load_validation_domains,
    run_domain_validation,
)


class DomainArgs:
    fs = 16000
    win_length = DEFAULT_WIN_LENGTH
    hop_length = DEFAULT_HOP_LENGTH
    n_fft = DEFAULT_N_FFT
    center = True
    segment_seconds = 4.0
    min_clean_rms_db = -40.0
    segment_attempts = 10
    valid_candidates = 16
    seed = 20260717
    identity_loss_weight = 0.0
    identity_energy_min_db = -50.0
    identity_energy_max_db = -20.0
    identity_gain_clamp_db = 20.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on every validation domain in a domains JSON."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-domains", required=True)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    domain_args = DomainArgs()
    domain_args.segment_seconds = args.segment_seconds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = GTCRN(nfft=config.get("n_fft", DEFAULT_N_FFT), fs=config.get("fs", 16000)).to(device)
    model.load_state_dict(checkpoint["model"])
    domain_args.n_fft = config.get("n_fft", DEFAULT_N_FFT)
    domain_args.fs = config.get("fs", 16000)
    domain_args.win_length = config.get("win_length", DEFAULT_WIN_LENGTH)
    domain_args.hop_length = config.get("hop_length", DEFAULT_HOP_LENGTH)
    domain_args.center = config.get("center", True)
    loss_fn = HybridLoss(
        domain_args.n_fft,
        domain_args.hop_length,
        domain_args.win_length,
        center=domain_args.center,
    ).to(device)

    domains = load_validation_domains(args.validation_domains)
    results = {}
    weighted_selection = 0.0
    selection_weight = 0.0
    all_gates_passed = True
    for domain in domains:
        dataset = build_domain_dataset(domain, domain_args)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        metrics = run_domain_validation(
            model, loader, loss_fn, device, domain_args, bool(domain.get("identity", False))
        )
        metrics["gate_passed"] = domain_gate_passed(metrics, domain.get("gate", {}))
        metrics["files"] = len(dataset)
        results[domain["name"]] = metrics
        weight = float(domain.get("weight", 1.0))
        weighted_selection += weight * domain_selection_value(metrics, domain)
        selection_weight += weight
        all_gates_passed = all_gates_passed and metrics["gate_passed"]
        print(
            f"{domain['name']}: loss={metrics['loss']:.4f} "
            f"si_snr={metrics['si_snr_db']:.2f}dB "
            f"pesq_change={metrics['pesq_change']:+.4f} "
            f"stoi_change={metrics['stoi_change']:+.5f} "
            f"gate={'pass' if metrics['gate_passed'] else 'fail'}"
        )
        sys.stdout.flush()

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_domains": str(Path(args.validation_domains).resolve()),
        "domains": results,
        "selection_loss": weighted_selection / max(selection_weight, 1e-12),
        "all_gates_passed": all_gates_passed,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()

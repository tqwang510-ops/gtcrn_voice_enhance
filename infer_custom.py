import argparse
import os
from pathlib import Path

import soundfile as sf
import torch

from audio_utils import enhance_waveform, read_wav
from gtcrn import GTCRN


def main():
    parser = argparse.ArgumentParser(description="Run GTCRN inference with a custom checkpoint.")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints_custom", "best.tar"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fs", type=int)
    parser.add_argument("--win-length", type=int)
    parser.add_argument("--hop-length", type=int)
    parser.add_argument("--n-fft", type=int)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    fs_expected = args.fs if args.fs is not None else config.get("fs", 16000)
    win_length = (
        args.win_length if args.win_length is not None else config.get("win_length", 160)
    )
    hop_length = (
        args.hop_length if args.hop_length is not None else config.get("hop_length", 80)
    )
    n_fft = args.n_fft if args.n_fft is not None else config.get("n_fft", 256)
    center = args.center if args.center is not None else config.get("center", True)

    model = GTCRN(nfft=n_fft, fs=fs_expected).to(device).eval()
    model.load_state_dict(ckpt["model"])

    mix, fs = read_wav(args.input, fs_expected)

    wav = torch.from_numpy(mix).to(device)
    with torch.no_grad():
        enhanced = enhance_waveform(model, wav, n_fft, hop_length, win_length, center=center)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, enhanced.detach().cpu().numpy(), fs_expected)


if __name__ == "__main__":
    main()

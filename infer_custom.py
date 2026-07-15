import argparse
import os

import soundfile as sf
import torch

from audio_utils import enhance_waveform, read_wav
from gtcrn import GTCRN


def main():
    parser = argparse.ArgumentParser(description="Run GTCRN inference with a custom checkpoint.")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints_custom", "best.tar"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    fs_expected = config.get("fs", 16000)
    win_length = config.get("win_length", 160)
    hop_length = config.get("hop_length", 80)
    n_fft = config.get("n_fft", 256)
    center = config.get("center", True)

    model = GTCRN(nfft=n_fft, fs=fs_expected).to(device).eval()
    model.load_state_dict(ckpt["model"])

    mix, fs = read_wav(args.input, fs_expected)

    wav = torch.from_numpy(mix).to(device)
    with torch.no_grad():
        enhanced = enhance_waveform(model, wav, n_fft, hop_length, win_length, center=center)
    sf.write(args.output, enhanced.detach().cpu().numpy(), fs_expected)


if __name__ == "__main__":
    main()

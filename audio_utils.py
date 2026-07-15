import numpy as np
import soundfile as sf
import torch


def read_wav(path, expected_fs=None):
    wav, fs = sf.read(path, dtype="float32")
    if expected_fs is not None and fs != expected_fs:
        raise ValueError(f"{path} sample rate is {fs}, expected {expected_fs}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, fs


def sqrt_hann_window(win_length, device):
    return torch.hann_window(win_length, device=device).pow(0.5)


def wav_to_stft(wav, n_fft, hop_length, win_length, center=True):
    spectrum = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=sqrt_hann_window(win_length, wav.device),
        center=center,
        return_complex=True,
    )
    return torch.view_as_real(spectrum)


def stft_to_wav(spec, n_fft, hop_length, win_length, length=None, center=True):
    spectrum = torch.view_as_complex(spec.contiguous())
    return torch.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=sqrt_hann_window(win_length, spec.device),
        center=center,
        length=length,
    )


def enhance_waveform(model, wav, n_fft, hop_length, win_length, center=True):
    spec = wav_to_stft(wav, n_fft, hop_length, win_length, center=center)
    enhanced_spec = model(spec[None])[0]
    return stft_to_wav(
        enhanced_spec,
        n_fft,
        hop_length,
        win_length,
        length=wav.shape[-1],
        center=center,
    )


def rms_dbfs(wav):
    wav = np.asarray(wav, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(wav * wav) + 1e-12) + 1e-12))


def si_snr_db(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    length = min(len(estimate), len(reference))
    estimate = estimate[:length] - np.mean(estimate[:length])
    reference = reference[:length] - np.mean(reference[:length])
    reference_energy = np.sum(reference * reference) + 1e-12
    target = np.sum(estimate * reference) * reference / reference_energy
    noise = estimate - target
    return float(10.0 * np.log10((np.sum(target * target) + 1e-12) / (np.sum(noise * noise) + 1e-12)))

"""Offline audio augmentations applied before W2V/MERT feature extraction."""

from __future__ import annotations
import random

import torch
import torchaudio.functional as F

from src.preprocessing.raw_boost import apply_rawboost


def aug_seed(stem: str, variant: int) -> int:
    return (variant * 1_000_003 + hash(stem) % (2**31 - 1)) & 0x7FFFFFFF


def augment_stem(stem: str, variant: int) -> str:
    """Insert aug tag before label so parse_split_label still works.

    train_foo_spoof -> train_foo_aug0_spoof
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return f"{stem}_aug{variant}"
    split, label = parts[0], parts[-1]
    middle = "_".join(parts[1:-1])
    if middle:
        return f"{split}_{middle}_aug{variant}_{label}"
    return f"{split}_aug{variant}_{label}"




def apply_audio_augment(wav: torch.Tensor, rng: random.Random, sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply robust audio forensics augmentations to mono waveform [1, T].
    Cel: Zniszczenie 'idealnych' warunków i symulacja kompresji, clippingu oraz zakłóceń OOD.
    """
    wav = wav.clone()

    # 0. RawBoost (LnL / ISD / SSI) — symulacja kanału transmisji ASVspoof
    if rng.random() < 0.35:
        wav_np = apply_rawboost(wav.squeeze(0).numpy(), sample_rate, rng)
        wav = torch.from_numpy(wav_np).unsqueeze(0)

    # 1. ZABÓJCA PASMA / Pseudo-Kompresja MP3 (Bandwidth bottleneck)
    # Drastycznie obniża częstotliwość próbkowania i wraca. 
    # Niszczy mikro-artefakty AI w górnych pasmach (10kHz+). Uczy model nie polegać tylko na "kryształowym" dźwięku.
    if rng.random() < 0.3:
        down_sr = rng.choice([4000, 6000, 8000]) # Symulacja jakości GSM / starego MP3
        wav = F.resample(wav, orig_freq=sample_rate, new_freq=down_sr)
        wav = F.resample(wav, orig_freq=down_sr, new_freq=sample_rate)

    # 2. CLIPPING / Przesterowanie (Symulacja taniego mikrofonu / Loudness War)
    # AI zazwyczaj ma idealną dynamikę. Przesterowanie wymusza uczenie się struktury, a nie głośności.
    if rng.random() < 0.2:
        gain = rng.uniform(1.5, 4.0)
        wav = torch.clamp(wav * gain, -1.0, 1.0) # Obcięcie wierzchołków fali (clipping)
        wav = wav / gain # Normalizacja z powrotem, by uniknąć eksplozji gradientów w MERT

    # 3. FILTR DOLNOPRZEPUSTOWY / Lowpass (Efekt ściany / telefonu)
    # Tnie wysokie częstotliwości. Uczy system radzenia sobie z przytłumionym audio.
    if rng.random() < 0.2:
        cutoff = rng.uniform(2000.0, 5000.0)
        wav = F.lowpass_biquad(wav, sample_rate, cutoff)

    # 4. SZUM ADDYTYWNY Z POPRAWNYM SNR (Signal-to-Noise Ratio)
    # Skalujemy szum inteligentnie do głośności aktualnej próbki (10dB do 25dB SNR).
    if rng.random() < 0.4:
        snr_db = rng.uniform(10.0, 25.0) # 10dB to mocny szum, 25dB to lekki szum w tle
        
        signal_power = wav.norm(p=2)
        noise = torch.randn_like(wav)
        noise_power = noise.norm(p=2)
        
        # Obliczenie mnożnika szumu ze wzoru na SNR
        if noise_power > 0:
            scale = signal_power / (noise_power * (10 ** (snr_db / 20.0)) + 1e-8)
            wav = wav + noise * scale

    # 5. ZMIANA GŁOŚNOŚCI (Volume jitter)
    if rng.random() < 0.5:
        wav = wav * rng.uniform(0.7, 1.2)

    return wav.clamp(-1.0, 1.0)
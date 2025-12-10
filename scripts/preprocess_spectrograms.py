import torch
import torchaudio
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse

def preprocess_dataset(input_dir, output_dir, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Transformacja (taka sama jak w modelu)
    transform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length
        ),
        torchaudio.transforms.AmplitudeToDB()
    )

    files = list(input_path.rglob("*.flac")) + list(input_path.rglob("*.wav")) # Dodaj inne rozszerzenia
    print(f"Found {len(files)} files to process.")

    for file_path in tqdm(files):
        try:
            # 1. Load
            wav, sr = torchaudio.load(file_path)
            
            # 2. Resample (jeśli trzeba)
            if sr != sample_rate:
                wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

            # 3. Mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 4. Compute MelSpec (cała długość!)
            mel_spec = transform(wav) # [1, n_mels, Time]

            # 5. Normalize (opcjonalnie, ale warto zapisać znormalizowane)
            # Uwaga: Normalizacja per-sample jest bezpieczna.
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            # 6. Transpose & Squeeze -> [Time, n_mels]
            # Mamba chce (Batch, Time, Features)
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)

            # 7. Save
            # Zachowujemy strukturę katalogów lub płasko z unikalną nazwą
            # Tutaj prościej: zachowujemy nazwę pliku, zmieniamy rozszerzenie na .pt
            rel_path = file_path.relative_to(input_path)
            save_path = output_path / rel_path.with_suffix(".pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(mel_spec.clone(), save_path) # clone() dla pewności pamięci

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Możesz tu dodać argparse
    preprocess_dataset(
        input_dir="/mnt/data/kusnierz/audio-data/SingFake/downloads_16k", # Twoje zresamplowane audio
        output_dir="/mnt/data/kusnierz/audio-data/SingFake/spectrograms",
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=80
    )
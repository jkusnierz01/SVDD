import hydra
from omegaconf import DictConfig
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig):

    input_dir = Path(cfg.preprocessing.input_dir)
    output_dir = Path(cfg.preprocessing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing on: {device}")

    if cfg.transform.mel_spec:
        transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.audio.sample_rate,
                n_fft=cfg.audio.n_fft,
                hop_length=cfg.audio.hop_length,
                n_mels=cfg.audio.n_mels
            )
        ).to(device)
    if cfg.transform.spec:
        transform = nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=cfg.audio.n_fft,
                hop_length=cfg.audio.hop_length,
            ),
        ).to(device)

    files = list(input_dir.rglob("*.flac"))
    
    with torch.no_grad():
        for file_path in tqdm(files, desc="Precomputing spectograms"):
            try:
                wav, sr = torchaudio.load(file_path)
                wav = wav.to(device)

                wav = wav / (wav.abs().max() + 1e-8)

                if sr != cfg.audio.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, cfg.audio.sample_rate).to(device)
                    wav = resampler(wav)


                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                spec = transform(wav) 
                spec = torch.clamp(spec, min=1e-6)
                spec = torch.log(spec)


                # spec = (spec - spec.mean()) / (spec.std() + 1e-6)

                spec = spec.squeeze(0).transpose(0, 1)

                save_path = output_dir / file_path.relative_to(input_dir).with_suffix(".pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(spec.cpu(), save_path)

            except Exception as e:
                print(f"Error {file_path}: {e}")

if __name__ == "__main__":
    main()
import hydra
from omegaconf import DictConfig
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
from src.preprocessing.base import BaseProcessor
from src.utils.preprocessing import pad_loop_torch


# longprocessor - taking long parts of tracks.
class SpectrogramsProcessor(BaseProcessor):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        device: str,
        sample_rate: int,
        total_len_sec: int,
        batch_size: int,
        n_fft: int,
        hop_length: int,
        n_mels: int = None,
        transform_type: str = "spectrogram",  # or "mel-spectrogram"
    ):
        self.device = device
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.total_len_sec = total_len_sec
        self.batch_size = batch_size
        self.transform_type = transform_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        
        if self.transform_type == "mel-spectrogram":
            self.transform = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                ),
            ).to(self.device)
            
        elif self.transform_type == "spectrogram":
            self.transform = nn.Sequential(
                torchaudio.transforms.Spectrogram(
                    n_fft=n_fft,
                    hop_length=hop_length,
                ),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def preprocess(self):
        print(f"Processing on device: {self.device}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = list(self.input_dir.rglob("*.flac"))
        
        target_samples = self.total_len_sec * self.sample_rate
        
        
        resamplers = {}
        for file in tqdm(files, desc="Processing files..."):
            try:
                wav, sr = torchaudio.load(file)
                
                if wav.shape[0] > 1:
                    # OR MEAN
                    # wav = torch.mean(wav, dim=0, keepdim=True)
                    channel_idx = torch.randint(0, wav.shape[0], (1,)).item()
                    wav = wav[channel_idx : channel_idx + 1]
                
                # 2. Resample
                if sr != self.sample_rate:
                    if sr not in resamplers:
                        resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    wav = resamplers[sr](wav)
                
                wav = pad_loop_torch(wav, target_samples)
                
                wav = wav.to(self.device)
                
                with torch.no_grad():
                    spec = self.transform(wav)
                    spec = torch.clamp(spec, min=1e-6)
                    spec = torch.log(spec)


                    spec = spec.squeeze(0).transpose(0, 1)

                save_path = self.output_dir / file.name
                save_path = save_path.with_suffix(".pt")
                
                torch.save(spec.cpu(), save_path)
            except Exception as e:
                print(f"Error {file}: {e}")
        
        


# @hydra.main(version_base=None, config_path="configs", config_name="preprocess")
# def main(cfg: DictConfig):

#     input_dir = Path(cfg.preprocessing.input_dir)
#     output_dir = Path(cfg.preprocessing.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Processing on: {device}")

#     if cfg.transform.mel_spec:
#         transform = nn.Sequential(
#             torchaudio.transforms.MelSpectrogram(
#                 sample_rate=cfg.audio.sample_rate,
#                 n_fft=cfg.audio.n_fft,
#                 hop_length=cfg.audio.hop_length,
#                 n_mels=cfg.audio.n_mels
#             )
#         ).to(device)
#     if cfg.transform.spec:
#         transform = nn.Sequential(
#             torchaudio.transforms.Spectrogram(
#                 n_fft=cfg.audio.n_fft,
#                 hop_length=cfg.audio.hop_length,
#             ),
#         ).to(device)

#     files = list(input_dir.rglob("*.flac"))

#     with torch.no_grad():
#         for file_path in tqdm(files, desc="Precomputing spectograms"):
#             try:
#                 wav, sr = torchaudio.load(file_path)
#                 wav = wav.to(device)

#                 wav = wav / (wav.abs().max() + 1e-8)

#                 if sr != cfg.audio.sample_rate:
#                     resampler = torchaudio.transforms.Resample(sr, cfg.audio.sample_rate).to(device)
#                     wav = resampler(wav)


#                 if wav.shape[0] > 1:
#                     wav = wav.mean(dim=0, keepdim=True)

#                 spec = transform(wav)
#                 spec = torch.clamp(spec, min=1e-6)
#                 spec = torch.log(spec)


#                 # spec = (spec - spec.mean()) / (spec.std() + 1e-6)

#                 spec = spec.squeeze(0).transpose(0, 1)

#                 save_path = output_dir / file_path.relative_to(input_dir).with_suffix(".pt")
#                 save_path.parent.mkdir(parents=True, exist_ok=True)
#                 torch.save(spec.cpu(), save_path)

#             except Exception as e:
#                 print(f"Error {file_path}: {e}")

# if __name__ == "__main__":
#     main()

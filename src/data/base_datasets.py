from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import librosa
import numpy as np
from torch import Tensor
import os
from pathlib import Path

def collate_fn(batch):
    features, labels = zip(*batch)

    lengths = torch.tensor([f.size(0) for f in features])
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    labels = torch.stack(labels)

    return features_padded, labels, lengths


class LongAudioDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        labels: list[int],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        max_len: int
    ):
        self.files = files
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_len = hop_length
        self.max_len = max_len

        # self.spec_transform = nn.Sequential(
        #     torchaudio.transforms.MelSpectrogram(
        #         sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_len
        #     ),
        #     torchaudio.transforms.AmplitudeToDB()
        # )

    def __getitem__(self, index):
        # wav, sr = torchaudio.load(self.files[index])
        # if wav.shape[0] > 1:
        #     wav = wav.mean(dim=0, keepdim=True)
        
        mel_spec = torch.load(self.files[index], weights_only=False)
        label = self.labels[index]

        # real_len = wav.shape[1]
        # if real_len < self.max_len:
        #     pad_amt = self.max_len - real_len
        #     wav = torch.nn.functional.pad(wav, (0, pad_amt))
        # else:
        #     wav = wav[:, : self.max_len]
        #     real_len = self.max_len

        # mel_spec = self.spec_transform(wav)
        # Odejmujemy średnią z CAŁEGO tensora, nie po wymiarze
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        # mel_spec = mel_spec.squeeze(0)
        # mel_spec = mel_spec.transpose(0, 1)
        
        return mel_spec, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.files)
    
    
class RawAudioDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    
    def __getitem__(self, index):
        pass
    
    
    def __len__(self):
        return len(self.files)
    
    
# code taken from SingFake repo
# https://github.com/yongyizang/SingFake/tree/main
# SingFake/models/feat+resnet/dataset.py
class SingFakeShortDataModule(Dataset):
    def __init__(
        self, data_dir: str, is_mixture: bool, target_sr:int, mode:str = "raw"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.is_mixture = is_mixture
        self.target_sr = target_sr
        self.cut = 64000
        self.mode = mode
        
        
        self.file_list = []
        data_path = Path(self.data_dir)
        if self.is_mixture:
            self.target_path = data_path / "mixtures"
        else:
            self.target_path = data_path / "vocals"
        
        assert self.target_path.exists(), f"{self.target_path} does not exist!"
        
        for file in os.listdir(self.target_path):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])

        # self.lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
        if self.mode == "spectrogram":
            self.spec_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=512, power=2, normalized=True)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        key = self.file_list[index]
        file_path = os.path.join(self.target_path, key + ".flac")
        # X, _ = sf.read(file_path, samplerate=self.target_sr)
        try:
            X, _ = librosa.load(file_path, sr=self.target_sr, mono=False)
            X = librosa.util.normalize(X)
        except Exception as e:
            print(f"Error loading {file_path}")
            return self.__getitem__(np.random.randint(len(self.file_list)))
        if X.shape[0] > 1:
            channel_id = np.random.randint(X.shape[0])
            X = X[channel_id]
        X_pad = pad_random(X, self.cut)
        sample = Tensor(X_pad)
        # x_inp = self.lfcc(xx_inp.unsqueeze(0))
        if self.mode == "spectrogram":
            sample = self.spec_transform(sample.unsqueeze(0))#.squeeze(0).transpose(0, 1)

        y = int(key.split("_")[0]) # TO BE CHECKED
        return sample, y
    
def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x 

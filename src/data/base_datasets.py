from torch.utils.data import Dataset
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence
import librosa
import numpy as np
from torch import Tensor
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
        max_len: int,
        **kwargs,
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


class SingFakeShortDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        target_sr: int,
        mode: str = "raw",
        **kwargs,
    ):
        super().__init__()
        self.files = files
        self.labels = labels
        self.target_sr = target_sr
        self.mode = mode
        self.cut = 64000
        
        self.n_fft = kwargs.get("n_fft", 512)
        self.hop_length = kwargs.get("hop_length", 160)
        self.win_length = kwargs.get("win_length", 512)

        if self.mode == "spectrogram":
            self.spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, power=2, normalized=True
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        y = self.labels[index]

        try:
            X, _ = librosa.load(str(file_path), sr=self.target_sr, mono=False)
            X = librosa.util.normalize(X)
        except Exception:
            return self.__getitem__(np.random.randint(len(self.files)))

        if X.ndim > 1 and X.shape[0] > 1:
            X = X[np.random.randint(X.shape[0])]

        X_pad = pad_random(X, self.cut)
        sample = Tensor(X_pad)

        if self.mode == "spectrogram":
            sample = self.spec_transform(sample.unsqueeze(0))
            sample = torch.transpose(sample, 1, 2)
            sample = sample.squeeze(0)

        return sample, torch.tensor(y, dtype=torch.long)


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len + 1)  # +1, aby high > 0
        return x[stt : stt + max_len]
    elif x_len < max_len:
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (num_repeats))[:max_len]
        return padded_x
    else:  # x_len == max_len
        return x

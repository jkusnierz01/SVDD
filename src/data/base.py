from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn as nn


class AudioDataset(Dataset):
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

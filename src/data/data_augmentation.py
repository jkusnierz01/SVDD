import torch
import torch.nn as nn
from torch import Tensor

class EmbeddingDataAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            LatentCutout(p=0.4),                
            TemporalDropout(p=0.5, max_ratio=0.7),
            FeatureDimDropout(p=0.4, max_ratio=0.4),
            LatentGaussianNoise(p=0.4, noise_scale=0.1), 
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class LatentGaussianNoise(nn.Module):
    def __init__(self, p: float = 0.5, noise_scale: float = 0.1):
        super().__init__()
        self.p = p
        self.noise_scale = noise_scale # np. 0.1 oznacza dodanie szumu o wielkości 10% aktualnego std

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            # Proporcjonalna siła szumu oparta na statystyce konkretnego pliku
            std = x.std() 
            x = x + torch.randn_like(x) * std * self.noise_scale
        return x


class LatentCutout(nn.Module):
    """Zeros out a random 2D block (time x features). Formerly named SpecAugment."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            time, embeds = x.shape
            t_len = torch.randint(10, 150, (1,)).item()
            e_len = torch.randint(10, 100, (1,)).item()
            t_start = torch.randint(0, max(1, time - t_len), (1,)).item()
            e_start = torch.randint(0, max(1, embeds - e_len), (1,)).item()
            
            x = x.clone()
            x[t_start:t_start + t_len, e_start:e_start + e_len] = 0
        return x


class TemporalDropout(nn.Module):
    """Zeros out a random contiguous window of time steps across all features."""
    def __init__(self, p: float = 0.3, max_ratio: float = 0.2):
        super().__init__()
        self.p = p
        self.max_ratio = max_ratio

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            time = x.shape[0]
            max_len = max(2, int(time * self.max_ratio))
            length = torch.randint(1, max_len, (1,)).item()
            start = torch.randint(0, max(1, time - length), (1,)).item()
            
            x = x.clone()
            x[start:start + length, :] = 0
        return x


class FeatureDimDropout(nn.Module):
    """Zeros out a random subset of feature dimensions across all time steps."""
    def __init__(self, p: float = 0.3, max_ratio: float = 0.15):
        super().__init__()
        self.p = p
        self.max_ratio = max_ratio

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            embeds = x.shape[1]
            max_dims = max(2, int(embeds * self.max_ratio))
            n_dims = torch.randint(1, max_dims, (1,)).item()
            
            # W PyTorch losowanie N unikalnych indeksów można zrobić przez permutację:
            dims = torch.randperm(embeds)[:n_dims]
            
            x = x.clone()
            x[:, dims] = 0
        return x
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class EmbeddingDataAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            SpecAugment(p=0.5),
            TemporalDropout(p=0.3),
            FeatureDimDropout(p=0.3),
            LatentGaussianNoise(p=0.5),
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class LatentGaussianNoise(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if np.random.uniform() < self.p:
            x = x + torch.randn_like(x) * 0.05
        return x


class SpecAugment(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if np.random.uniform() < self.p:
            time, embeds = x.shape
            t_len = np.random.randint(10, 150)
            e_len = np.random.randint(10, 100)
            t_start = np.random.randint(0, max(1, time - t_len))
            e_start = np.random.randint(0, embeds - e_len)
            x = x.clone()
            x[t_start:t_start + t_len, e_start:e_start + e_len] = 0
        return x


class TemporalDropout(nn.Module):
    """Zeros out a random contiguous window of time steps."""
    def __init__(self, p: float = 0.3, max_ratio: float = 0.2):
        super().__init__()
        self.p = p
        self.max_ratio = max_ratio

    def forward(self, x: Tensor) -> Tensor:
        if np.random.uniform() < self.p:
            time = x.shape[0]
            length = np.random.randint(1, max(2, int(time * self.max_ratio)))
            start = np.random.randint(0, max(1, time - length))
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
        if np.random.uniform() < self.p:
            embeds = x.shape[1]
            n_dims = np.random.randint(1, max(2, int(embeds * self.max_ratio)))
            dims = np.random.choice(embeds, n_dims, replace=False)
            x = x.clone()
            x[:, dims] = 0
        return x

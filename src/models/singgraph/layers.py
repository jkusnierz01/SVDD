"""Graph layers for SingGraph (ported from official SingGraph / AASIST backend)."""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.aasist import (
    GraphAttentionLayer,
    GraphPool,
    HtrgGraphAttentionLayer,
)

__all__ = [
    "GraphAttentionLayer",
    "GraphPool",
    "HtrgGraphAttentionLayer",
    "Residual_block",
]


class Residual_block(nn.Module):
    """RawNet2 residual block without extra max-pool (official SingGraph variant)."""

    def __init__(self, nb_filts: list[int], first: bool = False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out

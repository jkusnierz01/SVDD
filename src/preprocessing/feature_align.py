"""Linear temporal alignment for pre-extracted feature tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def align_time_axis_linear(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample/upsample features along the time dimension.

    Args:
        x: [T, D] or [B, T, D]
        target_len: target sequence length on the time axis

    Returns:
        Tensor with the same rank as input and time dim == target_len.
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

    t = x.shape[1]
    if t == target_len:
        return x.squeeze(0) if squeeze else x

    # [B, T, D] -> [B, D, T] for F.interpolate
    aligned = F.interpolate(
        x.float().transpose(1, 2),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return aligned.squeeze(0) if squeeze else aligned

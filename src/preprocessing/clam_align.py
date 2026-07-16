"""Temporal alignment of CLAM MERT features to W2V time axis."""

from __future__ import annotations

import torch
import torch.nn.functional as F

CLAM_NUM_LAYERS = 13
CLAM_LAYER_DIM = 768


def align_clam_time_axis(mert: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample/upsample MERT along time to match W2V length.

    Args:
        mert: [T_m, L, D] or [B, T_m, L, D]
        target_len: target time dimension (W2V seq len)

    Returns:
        Tensor with the same rank as input and time dim == target_len.
    """
    if mert.ndim == 3:
        mert = mert.unsqueeze(0)
        squeeze = True
    elif mert.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D MERT tensor, got shape {tuple(mert.shape)}")

    b, t_m, layers, dim = mert.shape
    if layers != CLAM_NUM_LAYERS or dim != CLAM_LAYER_DIM:
        raise ValueError(
            f"Expected [*, T, {CLAM_NUM_LAYERS}, {CLAM_LAYER_DIM}], got {tuple(mert.shape)}"
        )
    if t_m == target_len:
        return mert.squeeze(0) if squeeze else mert

    flat = mert.reshape(b, t_m, layers * dim).transpose(1, 2)  # [B, L*D, T_m]
    aligned = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
    out = aligned.transpose(1, 2).reshape(b, target_len, layers, dim)
    return out.squeeze(0) if squeeze else out

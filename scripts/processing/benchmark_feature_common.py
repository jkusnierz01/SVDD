"""Shared helpers for benchmark feature extraction (Proposed, CLAM, SingGraph)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.preprocessing.audio_augmentation import apply_audio_augment, aug_seed
from src.preprocessing.raw_boost import apply_rawboost

SONICS_BASE = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics")
DEFAULT_INPUT = SONICS_BASE / "benchmark_10pct" / "chunks_120s"
DEFAULT_OUTPUT = SONICS_BASE / "benchmark_10pct" / "features"
DEFAULT_DEMUCS_DIR = SONICS_BASE / "benchmark_10pct" / "demucs_120s"

SR = 16_000
MERT_SR = 24_000
PROPOSED_SEC = 120
PROPOSED_CHUNK_SEC = 30
CLAM_SEC = 90
DEMUCS_MODEL = "mdx_extra"
SINGGRAPH_WINDOW_SEC = 4
SINGGRAPH_WINDOW_SAMPLES = SINGGRAPH_WINDOW_SEC * SR
SINGGRAPH_NUM_WINDOWS = 30
# W2V XLS-R yields 1499 tokens/chunk × 4 = 5996; AvgPool(4,4) -> 1499 (same as precompute_pooled.py).
PROPOSED_RAW_SEQ_LEN = 5996
PROPOSED_POOLED_LEN = 1499
FEAT_DIM = 1024

W2V_XLSR = "facebook/wav2vec2-xls-r-300m"
MERT_330M = "m-a-p/MERT-v1-330M"
W2V_CLAM = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
MERT_95M = "m-a-p/MERT-v1-95M"

_POOL = nn.AvgPool1d(kernel_size=4, stride=4)


def read_chunk_flac(path: Path) -> torch.Tensor:
    """Read pre-prepared 16 kHz mono 120s FLAC -> [1, T]. No resample/mono."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"Expected sr={SR}, got {sr} for {path}")
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        raise ValueError(f"Expected mono chunk, got {wav.shape[0]} channels: {path}")
    return wav


def augment_chunk(wav: torch.Tensor, stem: str, aug_variant: int) -> torch.Tensor:
    rng = random.Random(aug_seed(stem, aug_variant))
    return apply_audio_augment(wav, rng, SR)


def clam_clip_90s(aug_wav: torch.Tensor) -> torch.Tensor:
    """First 90s of augmented 120s chunk."""
    return aug_wav[..., : CLAM_SEC * SR].contiguous()


def pool_proposed(seq: torch.Tensor) -> torch.Tensor:
    """[5996, 1024] -> [1499, 1024] via AvgPool1d(4,4) (matches SONICS precompute_pooled)."""
    if seq.ndim != 2:
        raise ValueError(f"Expected [seq, dim], got {tuple(seq.shape)}")
    return _POOL(seq.T.unsqueeze(0)).squeeze(0).T.contiguous()


def save_fp16(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().half(), path)


def load_demucs_stem(path: Path) -> torch.Tensor:
    """Load Demucs stem wav -> [1, T] mono 16 kHz float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = T.Resample(orig_freq=sr, new_freq=SR)(wav)
    return wav


def windowize_4s(wav: torch.Tensor) -> torch.Tensor:
    """[1, T] -> [N, 4*SR] non-overlapping 4s windows (drops trailing partial)."""
    window = SINGGRAPH_WINDOW_SAMPLES
    total = (wav.shape[-1] // window) * window
    if total < window:
        raise ValueError(f"Audio too short for 4s window: {wav.shape[-1]} samples")
    trimmed = wav[..., :total].squeeze(0)
    return trimmed.unfold(0, window, window)


def apply_rawboost_vocals(wav: torch.Tensor, stem: str, mode: str = "ssi") -> torch.Tensor:
    """RawBoost on vocals only. mode: 'ssi' (stationary) or 'full' (random alb 1-7)."""
    rng = random.Random(aug_seed(stem, 0))
    alb = 3 if mode == "ssi" else 8
    boosted = apply_rawboost(wav.squeeze(0).numpy(), SR, rng, alb=alb)
    return torch.from_numpy(boosted).unsqueeze(0)


def demucs_track_dir(demucs_dir: Path, stem: str) -> Path:
    return demucs_dir / DEMUCS_MODEL / stem


def demucs_vocals_path(demucs_dir: Path, stem: str) -> Path:
    return demucs_track_dir(demucs_dir, stem) / "vocals.wav"


def demucs_instrumental_path(demucs_dir: Path, stem: str) -> Path:
    return demucs_track_dir(demucs_dir, stem) / "no_vocals.wav"


@dataclass
class SingGraphPaths:
    vocals: Path
    instrumental: Path

    @classmethod
    def for_stem(cls, output_dir: Path, stem: str) -> SingGraphPaths:
        return cls(
            vocals=output_dir / "singgraph_features" / "vocals" / f"{stem}.pt",
            instrumental=output_dir / "singgraph_features" / "instrumental" / f"{stem}.pt",
        )

    def complete(self) -> bool:
        return self.vocals.exists() and self.instrumental.exists()

    def as_index_entry(self, stem: str) -> dict[str, str]:
        return {
            "stem": stem,
            "singgraph_vocals": str(self.vocals),
            "singgraph_instrumental": str(self.instrumental),
        }


def shard_files(files: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    if num_shards <= 1:
        return files
    return [f for i, f in enumerate(files) if i % num_shards == shard_id]


@dataclass
class OutputPaths:
    proposed_w2v: Path
    proposed_mert: Path
    proposed_pooled: Path
    clam_w2v: Path
    clam_mert: Path

    @classmethod
    def for_stem(cls, output_dir: Path, stem: str) -> OutputPaths:
        pooled_dir = output_dir.parent / "proposed_pooled"
        return cls(
            proposed_w2v=output_dir / "proposed_features" / "wav2vec" / f"{stem}.pt",
            proposed_mert=output_dir / "proposed_features" / "mert" / f"{stem}.pt",
            proposed_pooled=pooled_dir / f"{stem}.npy",
            clam_w2v=output_dir / "clam_features" / "wav2vec" / f"{stem}.pt",
            clam_mert=output_dir / "clam_features" / "mert" / f"{stem}.pt",
        )

    def proposed_complete(self) -> bool:
        return self.proposed_w2v.exists() and self.proposed_mert.exists()

    def proposed_pooled_complete(self) -> bool:
        return self.proposed_pooled.exists()

    def clam_complete(self) -> bool:
        return self.clam_w2v.exists() and self.clam_mert.exists()

    def all_complete(self) -> bool:
        return self.proposed_complete() and self.clam_complete()

    def should_skip(self, *, skip_proposed: bool, skip_clam: bool) -> bool:
        if skip_proposed and skip_clam:
            return True
        if skip_proposed:
            return self.clam_complete()
        if skip_clam:
            return self.proposed_complete()
        return self.all_complete()

    def as_index_entry(self, stem: str) -> dict[str, str]:
        entry = {
            "stem": stem,
            "proposed_w2v": str(self.proposed_w2v),
            "proposed_mert": str(self.proposed_mert),
            "clam_w2v": str(self.clam_w2v),
            "clam_mert": str(self.clam_mert),
        }
        if self.proposed_pooled.exists():
            entry["pooled"] = str(self.proposed_pooled)
        return entry


def merge_index_entry(index: list[dict], entry: dict[str, str]) -> None:
    stem = entry["stem"]
    for i, existing in enumerate(index):
        if existing.get("stem") == stem:
            index[i] = {**existing, **entry}
            return
    index.append(entry)


def index_entry_for_stem(output_dir: Path, stem: str) -> dict[str, str]:
    """Build index entry from files present on disk for one stem."""
    paths = OutputPaths.for_stem(output_dir, stem)
    sg = SingGraphPaths.for_stem(output_dir, stem)
    entry: dict[str, str] = {"stem": stem}

    if paths.proposed_w2v.exists() and paths.proposed_mert.exists():
        entry["proposed_w2v"] = str(paths.proposed_w2v.resolve())
        entry["proposed_mert"] = str(paths.proposed_mert.resolve())
    if paths.proposed_pooled.exists():
        entry["pooled"] = str(paths.proposed_pooled.resolve())
    if paths.clam_w2v.exists() and paths.clam_mert.exists():
        entry["clam_w2v"] = str(paths.clam_w2v.resolve())
        entry["clam_mert"] = str(paths.clam_mert.resolve())
    if sg.complete():
        entry["singgraph_vocals"] = str(sg.vocals.resolve())
        entry["singgraph_instrumental"] = str(sg.instrumental.resolve())

    return entry


def collect_feature_stems(features_dir: Path) -> set[str]:
    """Union of stems found under proposed/clam/singgraph feature folders."""
    stems: set[str] = set()
    for pattern in (
        "proposed_features/wav2vec/*.pt",
        "clam_features/wav2vec/*.pt",
        "singgraph_features/vocals/*.pt",
    ):
        for path in features_dir.glob(pattern):
            stems.add(path.stem)
    return stems


def load_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def save_index(index_path: Path, index: list[dict]) -> None:
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

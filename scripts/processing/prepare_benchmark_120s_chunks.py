#!/usr/bin/env python3
"""Pad/loop benchmark FLAC files to fixed 120s clips (base Sonics pipeline logic)."""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_augmentation import aug_seed

SONICS_BASE = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics")
DEFAULT_INPUT = SONICS_BASE / "benchmark_10pct" / "all_data_16k_mono"
DEFAULT_OUTPUT = SONICS_BASE / "benchmark_10pct" / "chunks_120s"

SR = 16_000
CHUNK_SEC = 120
CHUNK_SAMPLES = CHUNK_SEC * SR


def pad_loop_deterministic(wav: torch.Tensor, max_len: int, stem: str) -> torch.Tensor:
    """Same semantics as pad_loop_torch, but reproducible crop via aug_seed(stem, 0)."""
    x_len = wav.shape[-1]
    if x_len > max_len:
        rng = random.Random(aug_seed(stem, 0))
        start = rng.randint(0, x_len - max_len)
        return wav[..., start : start + max_len]
    if x_len < max_len:
        repeats = (max_len // x_len) + 1
        return wav.repeat(1, repeats)[..., :max_len]
    return wav


def load_mono_16k(path: Path) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = T.Resample(orig_freq=sr, new_freq=SR)(wav)
        sr = SR
    return wav, sr


def load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "files" in data:
        return data["files"]
    if isinstance(data, list):
        return data
    return []


def process_one_file(src: Path, output_dir: Path, skip_existing: bool) -> dict:
    """Worker: returns status dict (ok / skipped / error)."""
    stem = src.stem
    dst = output_dir / f"{stem}.flac"

    if skip_existing and dst.exists():
        return {"status": "skipped", "stem": stem}

    try:
        wav, _ = load_mono_16k(src)
        orig_samples = wav.shape[-1]
        clipped = pad_loop_deterministic(wav, CHUNK_SAMPLES, stem)
        sf.write(str(dst), clipped.squeeze(0).numpy(), SR, subtype="PCM_16")
        return {
            "status": "ok",
            "stem": stem,
            "entry": {
                "stem": stem,
                "source": str(src),
                "chunk_path": str(dst),
                "source_samples": orig_samples,
                "source_duration_sec": round(orig_samples / SR, 3),
                "chunk_samples": CHUNK_SAMPLES,
                "chunk_duration_sec": CHUNK_SEC,
            },
        }
    except Exception as exc:
        return {"status": "error", "stem": stem, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed 120s benchmark audio chunks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.flac"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"No FLAC files in {args.input_dir}", file=sys.stderr)
        return 1

    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_by_stem = {entry["stem"]: entry for entry in manifest}

    ok = skipped = errors = 0
    t0 = time.perf_counter()

    if args.dry_run:
        for src in files[:5]:
            print(f"Would write {args.output_dir / f'{src.stem}.flac'}")
        print(f"Dry run: {len(files)} files, workers={args.workers}")
        return 0

    pending = [
        src for src in files
        if not (args.skip_existing and (args.output_dir / f"{src.stem}.flac").exists())
    ]
    skipped = len(files) - len(pending)

    if not pending:
        print(f"All {len(files)} chunks already exist.")
        return 0

    print(f"Processing {len(pending)} files with {args.workers} workers ({skipped} skipped)")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_file, src, args.output_dir, False): src
            for src in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing 120s chunks"):
            result = future.result()
            status = result["status"]
            stem = result["stem"]
            if status == "ok":
                manifest_by_stem[stem] = result["entry"]
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"ERROR {stem}: {result.get('error')}", file=sys.stderr)

    manifest = sorted(manifest_by_stem.values(), key=lambda e: e["stem"])
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "sample_rate": SR,
        "chunk_sec": CHUNK_SEC,
        "crop_seed": "aug_seed(stem, 0)",
        "workers": args.workers,
        "files": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    rate = ok / elapsed if elapsed > 0 else 0.0
    print(
        f"Done: total={len(files)} ok={ok} skipped={skipped} errors={errors} "
        f"elapsed={elapsed:.1f}s ({rate:.1f} files/s) -> {args.output_dir}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Prepare deterministic 120s chunks for OOD datasets (M6/MoM).

Input layout:
  <input-root>/<class-name>/*.wav

Output:
  <output-dir>/<normalized_stem>.flac
  <output-dir>/manifest.json

Normalized stem format:
  test_<dataset>_<safe_source_stem>_<label>

Label mapping:
  - M6:  human -> bonafide, ai -> spoof
  - MoM: bonafide -> bonafide, deepfake -> deepfake
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm

SR = 16_000
CHUNK_SEC = 120
CHUNK_SAMPLES = CHUNK_SEC * SR

M6_LABEL_MAP = {"human": "bonafide", "ai": "spoof"}
MOM_LABEL_MAP = {"bonafide": "bonafide", "deepfake": "deepfake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed 120s OOD chunks for M6/MoM.")
    parser.add_argument("--dataset", type=str, required=True, choices=["m6", "mom"])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def label_map_for(dataset: str) -> dict[str, str]:
    if dataset == "m6":
        return M6_LABEL_MAP
    return MOM_LABEL_MAP


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned[:120] if cleaned else "unnamed"


def _short_hash(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]  # noqa: S324


def build_stem(dataset: str, source: Path, mapped_label: str) -> str:
    core = _safe_name(source.stem)
    return f"test_{dataset}_{core}_{_short_hash(source)}_{mapped_label}"


def load_audio_16k_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = T.Resample(orig_freq=sr, new_freq=SR)(wav)
    return wav


def crop_or_loop_120s(wav: torch.Tensor) -> torch.Tensor:
    n = wav.shape[-1]
    if n >= CHUNK_SAMPLES:
        return wav[..., :CHUNK_SAMPLES].contiguous()
    repeats = (CHUNK_SAMPLES // n) + 1
    return wav.repeat(1, repeats)[..., :CHUNK_SAMPLES].contiguous()


def discover_files(input_root: Path, label_map: dict[str, str]) -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    for src_label, mapped_label in label_map.items():
        class_dir = input_root / src_label
        if not class_dir.is_dir():
            continue
        for ext in ("*.wav", "*.flac", "*.mp3"):
            for path in sorted(class_dir.glob(ext)):
                rows.append((path, src_label, mapped_label))
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_map = label_map_for(args.dataset)
    discovered = discover_files(args.input_root, label_map)
    if args.limit is not None:
        discovered = discovered[: args.limit]

    if not discovered:
        print(f"No files found under {args.input_root}", file=sys.stderr)
        return 1

    if args.dry_run:
        for src, src_label, mapped_label in discovered[:5]:
            stem = build_stem(args.dataset, src, mapped_label)
            print(f"{src_label}: {src.name} -> {stem}.flac")
        print(f"Dry run: {len(discovered)} files")
        return 0

    manifest: list[dict] = []
    ok = skipped = errors = 0
    t0 = time.perf_counter()

    for src, src_label, mapped_label in tqdm(discovered, desc=f"Prepare {args.dataset} 120s"):
        stem = build_stem(args.dataset, src, mapped_label)
        dst = args.output_dir / f"{stem}.flac"
        try:
            if args.skip_existing and dst.exists():
                skipped += 1
            else:
                wav = load_audio_16k_mono(src)
                clipped = crop_or_loop_120s(wav)
                sf.write(str(dst), clipped.squeeze(0).numpy(), SR, subtype="PCM_16")
                ok += 1

            manifest.append(
                {
                    "stem": stem,
                    "dataset": args.dataset,
                    "source_label": src_label,
                    "mapped_label": mapped_label,
                    "source_path": str(src),
                    "chunk_path": str(dst),
                    "chunk_sec": CHUNK_SEC,
                    "sample_rate": SR,
                }
            )
        except Exception as exc:
            errors += 1
            print(f"ERROR {src}: {exc}", file=sys.stderr)

    manifest_path = args.output_dir / "manifest.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "chunk_sec": CHUNK_SEC,
        "sample_rate": SR,
        "total_discovered": len(discovered),
        "processed_ok": ok,
        "skipped": skipped,
        "errors": errors,
        "files": sorted(manifest, key=lambda x: x["stem"]),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    print(
        f"Done dataset={args.dataset} total={len(discovered)} ok={ok} skipped={skipped} "
        f"errors={errors} elapsed={elapsed:.1f}s output={args.output_dir}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Post-hoc align CLAM MERT time axis to W2V for existing benchmark features."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "processing"))

from benchmark_feature_common import (  # noqa: E402
    DEFAULT_OUTPUT,
    load_index,
    save_fp16,
    save_index,
)
from src.preprocessing.clam_align import (  # noqa: E402
    CLAM_LAYER_DIM,
    CLAM_NUM_LAYERS,
    align_clam_time_axis,
)


def _collect_stems(features_dir: Path) -> list[str]:
    index_path = features_dir / "index.json"
    if index_path.exists():
        data = load_index(index_path)
        stems = [
            e["stem"]
            for e in data
            if e.get("clam_w2v") and e.get("clam_mert")
        ]
        if stems:
            return sorted(set(stems))

    w2v_dir = features_dir / "clam_features" / "wav2vec"
    return sorted(p.stem for p in w2v_dir.glob("*.pt"))


def _paths_for_stem(features_dir: Path, stem: str) -> tuple[Path, Path]:
    w2v = features_dir / "clam_features" / "wav2vec" / f"{stem}.pt"
    mert = features_dir / "clam_features" / "mert" / f"{stem}.pt"
    return w2v, mert


def _process_stem(args: tuple[str, str, bool]) -> tuple[str, str]:
    stem, features_dir, skip_existing = args
    w2v_path, mert_path = _paths_for_stem(Path(features_dir), stem)

    if not w2v_path.is_file() or not mert_path.is_file():
        return stem, "missing_files"

    try:
        w2v = torch.load(w2v_path, map_location="cpu", weights_only=True)
        mert = torch.load(mert_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return stem, f"load_error: {exc}"

    if w2v.ndim != 3 or mert.ndim != 3:
        return stem, f"shape_error: w2v={tuple(w2v.shape)} mert={tuple(mert.shape)}"

    target_len = w2v.shape[0]
    if skip_existing and mert.shape[0] == target_len and w2v.shape == mert.shape:
        return stem, "skip"

    try:
        mert_aligned = align_clam_time_axis(mert.float(), target_len=target_len)
    except Exception as exc:
        return stem, f"align_error: {exc}"

    if mert_aligned.shape != w2v.shape:
        return stem, f"align_shape_error: w2v={tuple(w2v.shape)} mert={tuple(mert_aligned.shape)}"

    try:
        save_fp16(mert_aligned, mert_path)
    except Exception as exc:
        return stem, f"save_error: {exc}"

    return stem, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align CLAM MERT features to W2V time axis (offline, no GPU)."
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stems = _collect_stems(args.features_dir)
    if args.limit is not None:
        stems = stems[: args.limit]

    print(f"Features dir: {args.features_dir}")
    print(f"Stems:        {len(stems)}")
    print(f"Workers:      {args.workers}")

    if not stems:
        print("No CLAM stems found.")
        return 1

    if args.dry_run:
        for stem in stems[:5]:
            w2v, mert = _paths_for_stem(args.features_dir, stem)
            print(f"Would align {stem}: {mert.name} -> T from {w2v.name}")
        print(f"Dry run: {len(stems)} stems total.")
        return 0

    tasks = [(stem, str(args.features_dir), args.skip_existing) for stem in stems]
    ok = skipped = errors = 0
    error_samples: list[tuple[str, str]] = []

    with Pool(processes=args.workers) as pool:
        for i, (stem, status) in enumerate(pool.imap_unordered(_process_stem, tasks), 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                if len(error_samples) < 20:
                    error_samples.append((stem, status))

            if i % 1000 == 0 or i == len(stems):
                print(f"[{i}/{len(stems)}] ok={ok} skipped={skipped} errors={errors}")

    report = {
        "features_dir": str(args.features_dir),
        "total": len(stems),
        "aligned": ok,
        "skipped": skipped,
        "errors": errors,
        "expected_shape": f"[T, {CLAM_NUM_LAYERS}, {CLAM_LAYER_DIM}]",
    }
    print(json.dumps(report, indent=2))

    if error_samples:
        print("First errors:")
        for stem, msg in error_samples:
            print(f"  {stem}: {msg}")

    if errors:
        return 1

    print("Run rebuild_benchmark_features_index.py to refresh index.json if needed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

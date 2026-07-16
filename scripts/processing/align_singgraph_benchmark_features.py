#!/usr/bin/env python3
"""Post-hoc align SingGraph instrumental (MERT) time axis to vocals (W2V)."""

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
    FEAT_DIM,
    SINGGRAPH_NUM_WINDOWS,
    load_index,
    save_fp16,
)
from src.preprocessing.feature_align import align_time_axis_linear  # noqa: E402


def _collect_stems(features_dir: Path) -> list[str]:
    index_path = features_dir / "index.json"
    if index_path.exists():
        data = load_index(index_path)
        stems = [
            e["stem"]
            for e in data
            if e.get("singgraph_vocals") and e.get("singgraph_instrumental")
        ]
        if stems:
            return sorted(set(stems))

    vocals_dir = features_dir / "singgraph_features" / "vocals"
    return sorted(p.stem for p in vocals_dir.glob("*.pt"))


def _paths_for_stem(features_dir: Path, stem: str) -> tuple[Path, Path]:
    vocals = features_dir / "singgraph_features" / "vocals" / f"{stem}.pt"
    instrumental = features_dir / "singgraph_features" / "instrumental" / f"{stem}.pt"
    return vocals, instrumental


def _process_stem(args: tuple[str, str, bool]) -> tuple[str, str]:
    stem, features_dir, skip_existing = args
    vocals_path, inst_path = _paths_for_stem(Path(features_dir), stem)

    if not vocals_path.is_file() or not inst_path.is_file():
        return stem, "missing_files"

    try:
        vocals = torch.load(vocals_path, map_location="cpu", weights_only=True)
        instrumental = torch.load(inst_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return stem, f"load_error: {exc}"

    if vocals.ndim != 3 or instrumental.ndim != 3:
        return stem, f"shape_error: vocals={tuple(vocals.shape)} inst={tuple(instrumental.shape)}"
    if vocals.shape[0] != SINGGRAPH_NUM_WINDOWS or instrumental.shape[0] != SINGGRAPH_NUM_WINDOWS:
        return stem, (
            f"windows_error: vocals={tuple(vocals.shape)} inst={tuple(instrumental.shape)} "
            f"expected dim0={SINGGRAPH_NUM_WINDOWS}"
        )
    if vocals.shape[2] != FEAT_DIM or instrumental.shape[2] != FEAT_DIM:
        return stem, f"dim_error: vocals={tuple(vocals.shape)} inst={tuple(instrumental.shape)}"

    target_len = vocals.shape[1]
    if skip_existing and instrumental.shape == vocals.shape:
        return stem, "skip"

    try:
        inst_aligned = align_time_axis_linear(instrumental, target_len=target_len)
    except Exception as exc:
        return stem, f"align_error: {exc}"

    if inst_aligned.shape != vocals.shape:
        return stem, (
            f"align_shape_error: vocals={tuple(vocals.shape)} inst={tuple(inst_aligned.shape)}"
        )

    try:
        save_fp16(inst_aligned, inst_path)
    except Exception as exc:
        return stem, f"save_error: {exc}"

    return stem, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align SingGraph instrumental (MERT) to vocals (W2V) time axis."
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
        print("No SingGraph stems found.")
        return 1

    if args.dry_run:
        for stem in stems[:5]:
            vocals, inst = _paths_for_stem(args.features_dir, stem)
            print(f"Would align {stem}: {inst.name} -> T from {vocals.name}")
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
        "expected_shape": f"[{SINGGRAPH_NUM_WINDOWS}, T, {FEAT_DIM}]",
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

#!/usr/bin/env python3
"""Merge benchmark Proposed W2V + MERT .pt files into pooled .npy for Mamba training.

Input (per stem):
  proposed_features/wav2vec/{stem}.pt  -> [1499, 1024] fp16
  proposed_features/mert/{stem}.pt     -> [1499, 1024] fp16

Output (per stem):
  proposed_pooled/{stem}.npy             -> [1499, 2048] float16

Also writes proposed_pooled/index.json with entries: {stem, pooled}.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "processing"))

from benchmark_feature_common import (  # noqa: E402
    DEFAULT_OUTPUT,
    FEAT_DIM,
    PROPOSED_POOLED_LEN,
    SONICS_BASE,
)

DEFAULT_FEATURES_DIR = DEFAULT_OUTPUT
DEFAULT_MERGE_OUTPUT = SONICS_BASE / "benchmark_10pct" / "proposed_pooled"


def _resolve_paths(item: dict, features_dir: Path) -> tuple[str, Path, Path] | None:
    stem = item.get("stem")
    if not stem:
        return None

    w2v = item.get("proposed_w2v")
    mert = item.get("proposed_mert")
    if w2v and mert:
        return stem, Path(w2v), Path(mert)

    w2v_path = features_dir / "proposed_features" / "wav2vec" / f"{stem}.pt"
    mert_path = features_dir / "proposed_features" / "mert" / f"{stem}.pt"
    if w2v_path.exists() and mert_path.exists():
        return stem, w2v_path, mert_path
    return None


def _collect_items(features_dir: Path) -> list[dict]:
    index_path = features_dir / "index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    items: list[dict] = []
    seen: set[str] = set()

    for entry in data:
        resolved = _resolve_paths(entry, features_dir)
        if resolved is None:
            continue
        stem, w2v_path, mert_path = resolved
        if stem in seen:
            continue
        seen.add(stem)
        items.append({"stem": stem, "proposed_w2v": str(w2v_path), "proposed_mert": str(mert_path)})

    w2v_dir = features_dir / "proposed_features" / "wav2vec"
    if w2v_dir.is_dir():
        for w2v_path in sorted(w2v_dir.glob("*.pt")):
            stem = w2v_path.stem
            if stem in seen:
                continue
            mert_path = features_dir / "proposed_features" / "mert" / f"{stem}.pt"
            if mert_path.exists():
                seen.add(stem)
                items.append(
                    {
                        "stem": stem,
                        "proposed_w2v": str(w2v_path),
                        "proposed_mert": str(mert_path),
                    }
                )

    return items


def _is_valid_pooled_npy(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return np.load(path).size > 0
    except Exception:
        return False


def _process_item(args: tuple[dict, str, bool]) -> tuple[str, str]:
    item, output_dir, skip_existing = args
    stem = item["stem"]
    out_path = Path(output_dir) / f"{stem}.npy"

    if skip_existing and _is_valid_pooled_npy(out_path):
        return stem, "skip"

    try:
        w2v = torch.load(item["proposed_w2v"], weights_only=True).float()
        mert = torch.load(item["proposed_mert"], weights_only=True).float()
    except Exception as exc:
        return stem, f"load_error: {exc}"

    if w2v.ndim != 2 or mert.ndim != 2:
        return stem, f"shape_error: w2v={tuple(w2v.shape)} mert={tuple(mert.shape)}"
    if w2v.shape[1] != FEAT_DIM or mert.shape[1] != FEAT_DIM:
        return stem, f"dim_error: w2v={tuple(w2v.shape)} mert={tuple(mert.shape)}"
    if w2v.shape[0] != mert.shape[0]:
        return stem, f"seq_mismatch: w2v={w2v.shape[0]} mert={mert.shape[0]}"
    if w2v.shape[0] != PROPOSED_POOLED_LEN:
        pass  # warn only; still merge

    pooled = torch.cat([w2v, mert], dim=1)
    if pooled.shape != (w2v.shape[0], 2 * FEAT_DIM):
        return stem, f"concat_error: {tuple(pooled.shape)}"

    try:
        np.save(str(out_path), pooled.half().numpy())
    except Exception as exc:
        return stem, f"save_error: {exc}"

    return stem, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge benchmark Proposed W2V+MERT into pooled .npy for training."
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MERGE_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = _collect_items(args.features_dir)

    print(f"Features dir: {args.features_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Entries:      {len(items)}")
    print(f"Workers:      {args.workers}")

    if not items:
        print("No proposed W2V+MERT pairs found.")
        return 1

    if args.dry_run:
        for item in items[:5]:
            out_path = args.output_dir / f"{item['stem']}.npy"
            print(f"Would merge {item['stem']} -> {out_path}")
        print(f"Dry run: {len(items)} stems total.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(item, str(args.output_dir), args.skip_existing) for item in items]

    new_index: list[dict] = []
    errors: list[tuple[str, str]] = []
    skipped = done = 0

    with Pool(processes=args.workers) as pool:
        for i, (stem, status) in enumerate(pool.imap_unordered(_process_item, tasks), 1):
            if status == "ok":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                errors.append((stem, status))

            out_path = args.output_dir / f"{stem}.npy"
            if out_path.exists():
                new_index.append({"stem": stem, "pooled": str(out_path.resolve())})

            if i % 1000 == 0 or i == len(items):
                print(f"[{i}/{len(items)}]  done={done}  skipped={skipped}  errors={len(errors)}")

    index_out = args.output_dir / "index.json"
    with open(index_out, "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)

    report = {
        "features_dir": str(args.features_dir),
        "output_dir": str(args.output_dir),
        "total": len(items),
        "processed": done,
        "skipped": skipped,
        "errors": len(errors),
        "index_entries": len(new_index),
    }
    print(json.dumps(report, indent=2))

    if errors:
        print("First 20 errors:")
        for stem, msg in errors[:20]:
            print(f"  {stem}: {msg}")
        return 1

    print(f"Index written to: {index_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

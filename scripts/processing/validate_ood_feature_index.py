#!/usr/bin/env python3
"""Minimal sanity checks for OOD CLAM/SingGraph feature index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OOD feature index and tensor shapes.")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["m6", "mom"])
    parser.add_argument("--check-clam", action="store_true")
    parser.add_argument("--check-singgraph", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _expected_stem_prefix(dataset: str) -> str:
    return f"test_{dataset}_"


def _check_clam(entry: dict) -> str | None:
    w2v = Path(entry["clam_w2v"])
    mert = Path(entry["clam_mert"])
    if not w2v.exists() or not mert.exists():
        return "missing_clam_files"

    w = torch.load(w2v, map_location="cpu", weights_only=True)
    m = torch.load(mert, map_location="cpu", weights_only=True)
    if w.ndim != 3 or m.ndim != 3:
        return f"clam_rank_error w={tuple(w.shape)} m={tuple(m.shape)}"
    if w.shape[1:] != (13, 768):
        return f"clam_w2v_shape_error w={tuple(w.shape)}"
    if m.shape != w.shape:
        return f"clam_mismatch w={tuple(w.shape)} m={tuple(m.shape)}"
    return None


def _check_singgraph(entry: dict) -> str | None:
    vpath = Path(entry["singgraph_vocals"])
    ipath = Path(entry["singgraph_instrumental"])
    if not vpath.exists() or not ipath.exists():
        return "missing_singgraph_files"

    v = torch.load(vpath, map_location="cpu", weights_only=True)
    i = torch.load(ipath, map_location="cpu", weights_only=True)
    if v.ndim != 3 or i.ndim != 3:
        return f"singgraph_rank_error v={tuple(v.shape)} i={tuple(i.shape)}"
    if v.shape[0] != 30 or v.shape[2] != 1024:
        return f"singgraph_v_shape_error v={tuple(v.shape)}"
    if i.shape != v.shape:
        return f"singgraph_mismatch v={tuple(v.shape)} i={tuple(i.shape)}"
    return None


def main() -> int:
    args = parse_args()
    if not args.check_clam and not args.check_singgraph:
        args.check_clam = True
        args.check_singgraph = True

    index_path = args.features_dir / "index.json"
    if not index_path.exists():
        print(f"Missing {index_path}", file=sys.stderr)
        return 1

    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    expected_prefix = _expected_stem_prefix(args.dataset)
    errors: list[tuple[str, str]] = []
    checked = 0

    for entry in data:
        stem = entry.get("stem", "")
        if not stem.startswith(expected_prefix):
            continue

        if args.check_clam and entry.get("clam_w2v") and entry.get("clam_mert"):
            err = _check_clam(entry)
            checked += 1
            if err:
                errors.append((stem, err))

        if args.check_singgraph and entry.get("singgraph_vocals") and entry.get("singgraph_instrumental"):
            err = _check_singgraph(entry)
            checked += 1
            if err:
                errors.append((stem, err))

        if args.limit is not None and checked >= args.limit:
            break

    report = {
        "features_dir": str(args.features_dir),
        "dataset": args.dataset,
        "checked_items": checked,
        "errors": len(errors),
    }
    print(json.dumps(report, indent=2))
    if errors:
        print("First errors:")
        for stem, msg in errors[:20]:
            print(f"  {stem}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Merge clean and train-aug feature indexes.

By default:
- keeps all entries from clean index,
- appends only `train_*` entries from aug indexes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _is_train(stem: str) -> bool:
    return stem.startswith("train_")


def _sig(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge clean + train-aug feature indexes.")
    parser.add_argument("--clean", type=Path, required=True, help="Clean index.json")
    parser.add_argument("--aug", type=Path, nargs="+", required=True, help="Augmented index.json files")
    parser.add_argument("--out", type=Path, required=True, help="Output merged index.json")
    parser.add_argument(
        "--include-non-train-aug",
        action="store_true",
        help="Include valid/test aug entries too (default: skip).",
    )
    args = parser.parse_args()

    merged = list(_load(args.clean))
    seen = {_sig(e) for e in merged}
    added = skipped = 0

    for aug_path in args.aug:
        for entry in _load(aug_path):
            stem = str(entry.get("stem", ""))
            if not stem:
                skipped += 1
                continue
            if not args.include_non_train_aug and not _is_train(stem):
                skipped += 1
                continue
            signature = _sig(entry)
            if signature in seen:
                skipped += 1
                continue
            merged.append(entry)
            seen.add(signature)
            added += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "clean_entries": len(_load(args.clean)),
                "merged_entries": len(merged),
                "aug_added": added,
                "aug_skipped": skipped,
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

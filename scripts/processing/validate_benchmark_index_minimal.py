#!/usr/bin/env python3
"""Minimal benchmark index sanity checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal benchmark index checks.")
    parser.add_argument("--index", type=Path, required=True, help="Path to index.json")
    parser.add_argument(
        "--forbid-aug-outside-train",
        action="store_true",
        help="Fail when stem contains '_aug' and split is not train.",
    )
    args = parser.parse_args()

    with open(args.index, encoding="utf-8") as f:
        data = json.load(f)

    splits = Counter()
    bad_aug = []
    for entry in data:
        stem = str(entry.get("stem", ""))
        if not stem:
            continue
        split = stem.split("_", 1)[0]
        splits[split] += 1
        if args.forbid_aug_outside_train and "_aug" in stem and split != "train":
            bad_aug.append(stem)

    print(json.dumps({"entries": len(data), "split_counts": dict(splits)}, indent=2))

    if bad_aug:
        print("Invalid augmented stems outside train (first 20):", file=sys.stderr)
        for stem in bad_aug[:20]:
            print(f"  {stem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

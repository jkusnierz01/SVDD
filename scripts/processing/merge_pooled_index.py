"""
Merge clean and augmented pooled index.json files for training.

Typical use:
  - clean index: all splits (train, valid, test) from preprocessed_pooled_fp16
  - aug index:   train-only augmented variants from preprocessed_pooled_aug_v0
  - output:      merged index pointing to both (paths unchanged on disk)

Valid/test entries always come from the clean index only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_index(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def is_train_stem(stem: str) -> bool:
    return stem.split("_", 1)[0] == "train"


def merge(
    clean_index: Path,
    aug_indexes: list[Path],
    output: Path,
    train_only_from_aug: bool = True,
) -> dict:
    clean = load_index(clean_index)
    merged: list[dict] = list(clean)
    seen_pooled = {e.get("pooled") for e in clean if e.get("pooled")}

    aug_added = 0
    aug_skipped = 0

    for aug_path in aug_indexes:
        for entry in load_index(aug_path):
            stem = entry.get("stem", "")
            pooled = entry.get("pooled")
            if not stem or not pooled:
                aug_skipped += 1
                continue
            if train_only_from_aug and not is_train_stem(stem):
                aug_skipped += 1
                continue
            if pooled in seen_pooled:
                aug_skipped += 1
                continue
            merged.append(entry)
            seen_pooled.add(pooled)
            aug_added += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    n_train_clean = sum(1 for e in clean if is_train_stem(e.get("stem", "")))
    n_train_merged = sum(1 for e in merged if is_train_stem(e.get("stem", "")))

    return {
        "clean_entries": len(clean),
        "merged_entries": len(merged),
        "aug_added": aug_added,
        "aug_skipped": aug_skipped,
        "train_clean": n_train_clean,
        "train_merged": n_train_merged,
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description="Merge clean + aug pooled index.json")
    parser.add_argument("--clean", required=True, type=Path, help="Clean pooled index.json")
    parser.add_argument(
        "--aug",
        required=True,
        type=Path,
        nargs="+",
        help="One or more aug pooled index.json files (train-only entries kept)",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output merged index.json")
    parser.add_argument(
        "--include-non-train-aug",
        action="store_true",
        help="Also include valid/test from aug indexes (default: skip)",
    )
    args = parser.parse_args()

    stats = merge(
        clean_index=args.clean,
        aug_indexes=args.aug,
        output=args.out,
        train_only_from_aug=not args.include_non_train_aug,
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

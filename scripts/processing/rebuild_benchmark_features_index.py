#!/usr/bin/env python3
"""Rebuild features/index.json by scanning feature files on disk.

Fixes incomplete index from parallel SLURM shards overwriting each other.
Scans proposed_features/, clam_features/, singgraph_features/ and merges
all available paths per stem into a single index entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "processing"))

from benchmark_feature_common import (  # noqa: E402
    DEFAULT_OUTPUT,
    collect_feature_stems,
    index_entry_for_stem,
    save_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild benchmark features index.json from disk.")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true", help="Copy existing index.json to index.json.bak")
    return parser.parse_args()


def _count_keys(index: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in index:
        for key in entry:
            if key == "stem":
                continue
            counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    features_dir = args.features_dir
    index_path = features_dir / "index.json"

    stems = sorted(collect_feature_stems(features_dir))
    if not stems:
        print(f"No feature files under {features_dir}", file=sys.stderr)
        return 1

    index: list[dict] = []
    skipped = 0
    for stem in stems:
        entry = index_entry_for_stem(features_dir, stem)
        if len(entry) <= 1:
            skipped += 1
            continue
        index.append(entry)

    counts = _count_keys(index)
    report = {
        "features_dir": str(features_dir),
        "stems_on_disk": len(stems),
        "index_entries": len(index),
        "skipped_empty": skipped,
        "key_counts": counts,
    }

    print(json.dumps(report, indent=2))

    if args.dry_run:
        print(f"Dry run: would write {len(index)} entries to {index_path}")
        return 0

    if args.backup and index_path.exists():
        backup_path = index_path.with_suffix(".json.bak")
        backup_path.write_bytes(index_path.read_bytes())
        print(f"Backed up existing index to {backup_path}")

    save_index(index_path, index)
    print(f"Wrote {index_path} ({len(index)} entries)")

    splits = Counter(e["stem"].split("_")[0] for e in index)
    print("Split distribution:", dict(splits))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Check whether SONICS training indexes include enlarged train (clean + aug)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset import parse_split_label


def _load_index(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _index_stats(data: list[dict], *, path_key: str | None = None) -> dict:
    split_counts = Counter()
    train_stems = Counter()
    train_paths: set[str] = set()

    for entry in data:
        stem = str(entry.get("stem", ""))
        if not stem:
            continue
        split, _ = parse_split_label(stem)
        split_counts[split] += 1
        if split == "train":
            train_stems[stem] += 1
            if path_key and entry.get(path_key):
                train_paths.add(str(entry[path_key]))

    duplicate_train_stems = sum(1 for c in train_stems.values() if c > 1)
    return {
        "entries": len(data),
        "split_counts": dict(split_counts),
        "train_entries": split_counts.get("train", 0),
        "unique_train_stems": len(train_stems),
        "duplicate_train_stems": duplicate_train_stems,
        "unique_train_paths": len(train_paths) if path_key else None,
    }


def _check_loader(name: str, module_path: str, data_dir: Path) -> dict:
    import importlib

    if not data_dir.is_dir():
        return {"name": name, "error": f"missing data_dir: {data_dir}"}

    mod_name, cls_name = module_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    dm = cls(data_dir=str(data_dir), batch_size=2, num_workers=0, transform=None)
    dm.setup("fit")
    train_n = len(dm.training_dataset)
    valid_n = len(dm.valid_dataset)
    test_n = len(dm.test_dataset)
    return {
        "name": name,
        "data_dir": str(data_dir),
        "train": train_n,
        "valid": valid_n,
        "test": test_n,
        "enlarged_train": train_n > 6406,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate enlarged SONICS training data.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct"),
    )
    parser.add_argument(
        "--expected-clean-train",
        type=int,
        default=6406,
        help="Train count expected from clean-only benchmark subset.",
    )
    parser.add_argument("--check-loaders", action="store_true", help="Instantiate dataloaders.")
    args = parser.parse_args()

    base = args.base
    features_index = base / "features_clean" / "index.json"
    pooled_index = base / "proposed_pooled" / "index.json"

    report: dict = {"base": str(base), "indexes": {}, "loaders": [], "pass": True}

    for label, path, path_key in [
        ("features_clean", features_index, "clam_w2v"),
        ("proposed_pooled", pooled_index, "pooled"),
    ]:
        if not path.exists():
            report["indexes"][label] = {"error": f"missing {path}"}
            report["pass"] = False
            continue
        stats = _index_stats(_load_index(path), path_key=path_key)
        stats["enlarged_train"] = stats["train_entries"] > args.expected_clean_train
        stats["expected_clean_train"] = args.expected_clean_train
        report["indexes"][label] = stats
        if not stats["enlarged_train"]:
            report["pass"] = False

    if args.check_loaders:
        for name, cls, data_dir in [
            ("mamba", "src.data.sonics_dataloader.SonicsDataModule", base / "proposed_pooled"),
            ("clam", "src.data.clam_dataloader.ClamDataModule", base / "features_clean"),
            ("singgraph", "src.data.singgraph_dataloader.SingGraphDataModule", base / "features_clean"),
        ]:
            try:
                loader_report = _check_loader(name, cls, data_dir)
            except Exception as exc:
                loader_report = {"name": name, "error": str(exc)}
                report["pass"] = False
            if loader_report.get("train") is not None and not loader_report.get("enlarged_train"):
                report["pass"] = False
            report["loaders"].append(loader_report)

    print(json.dumps(report, indent=2))

    if report["pass"]:
        print("\nPASS: train looks enlarged (clean + aug).")
        return 0

    print(
        "\nFAIL: train still looks clean-only. Run:\n"
        "  sbatch scripts/bash/finalize_sonics_benchmark_indices.sh",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

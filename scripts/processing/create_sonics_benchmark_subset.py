#!/usr/bin/env python3
"""Create a stratified benchmark subset of the SONICS dataset."""

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

SONICS_BASE = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics")
DEFAULT_METADATA = SONICS_BASE / "all_data" / "metadata.json"
DEFAULT_AUDIO_DIR = SONICS_BASE / "all_data_16k_mono"
DEFAULT_OUTPUT_DIR = SONICS_BASE / "benchmark_10pct"

PROPORTION_TOLERANCE_PCT = 0.01


def strat_key(row: pd.Series) -> tuple[str, str, str]:
    if row["spoof"] == "bonafide":
        return (row["set"], "bonafide", "bonafide")
    return (row["set"], "deepfake", row["label"])


def format_stratum(key: tuple[str, str, str]) -> str:
    return " | ".join(key)


def compute_sample_counts(
    counts: dict[tuple[str, str, str], int],
    fraction: float,
) -> dict[tuple[str, str, str], int]:
    """Hamilton/largest-remainder allocation for exact target total."""
    quotas = {k: v * fraction for k, v in counts.items()}
    n_take = {k: int(math.floor(q)) for k, q in quotas.items()}
    target_total = round(sum(counts.values()) * fraction)
    remainder = target_total - sum(n_take.values())

    if remainder > 0:
        ranked = sorted(
            counts.keys(),
            key=lambda k: (quotas[k] - n_take[k], counts[k], format_stratum(k)),
            reverse=True,
        )
        for key in ranked[:remainder]:
            n_take[key] += 1

    return n_take


def select_subset(
    df: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    df = df.copy()
    df["stratum"] = df.apply(strat_key, axis=1)

    counts = df.groupby("stratum", sort=False).size().to_dict()
    n_take = compute_sample_counts(counts, fraction)

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for stratum, take in sorted(n_take.items(), key=lambda x: format_stratum(x[0])):
        stratum_df = df[df["stratum"] == stratum].sort_values("filename")
        if take > len(stratum_df):
            raise ValueError(
                f"Stratum {format_stratum(stratum)}: requested {take} > available {len(stratum_df)}"
            )
        if take == 0:
            continue
        chosen = rng.choice(len(stratum_df), size=take, replace=False)
        selected_indices.extend(stratum_df.iloc[chosen].index.tolist())

    return df.loc[sorted(selected_indices)].reset_index(drop=True)


def _distribution_table(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    full_total = len(full_df)
    subset_total = len(subset_df)

    full_counts = full_df.groupby(group_cols, sort=False).size()
    subset_counts = subset_df.groupby(group_cols, sort=False).size()

    rows = []
    for key in full_counts.index:
        full_n = int(full_counts.loc[key])
        subset_n = int(subset_counts.get(key, 0))
        full_pct = 100.0 * full_n / full_total
        subset_pct = 100.0 * subset_n / subset_total if subset_total else 0.0
        label = key if isinstance(key, str) else " | ".join(key)
        rows.append(
            {
                "group": label,
                "full": full_n,
                "subset": subset_n,
                "full_pct": full_pct,
                "subset_pct": subset_pct,
                "delta_pct": subset_pct - full_pct,
            }
        )

    return pd.DataFrame(rows)


def build_report(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    fraction: float,
    seed: int,
    metadata_path: Path,
    audio_dir: Path,
    output_dir: Path,
    elapsed_s: float,
    copied: int,
    skipped: int,
    dry_run: bool,
) -> tuple[str, float]:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SONICS BENCHMARK SUBSET — STRATIFICATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Metadata:        {metadata_path}")
    lines.append(f"Audio source:    {audio_dir}")
    lines.append(f"Output:          {output_dir}")
    lines.append(f"Fraction:        {fraction:.4f}")
    lines.append(f"Seed:            {seed}")
    lines.append(f"Dry run:         {dry_run}")
    lines.append(f"Elapsed:         {elapsed_s:.2f}s")
    lines.append(f"Copied files:    {copied}")
    lines.append(f"Skipped files:   {skipped}")
    lines.append("")

    full_total = len(full_df)
    subset_total = len(subset_df)
    lines.append(f"TOTAL: full={full_total:,}  subset={subset_total:,}  "
                 f"({100.0 * subset_total / full_total:.2f}% of full)")
    lines.append("")

    stratum_full = full_df.copy()
    stratum_full["stratum"] = stratum_full.apply(strat_key, axis=1)
    stratum_subset = subset_df.copy()
    stratum_full["stratum_label"] = stratum_full["stratum"].map(format_stratum)
    stratum_subset["stratum_label"] = stratum_subset["stratum"].map(format_stratum)

    strata_table = _distribution_table(stratum_full, stratum_subset, ["stratum_label"])
    strata_table = strata_table.rename(
        columns={
            "group": "stratum",
            "full": "full_n",
            "subset": "subset_n",
            "full_pct": "full_%",
            "subset_pct": "subset_%",
            "delta_pct": "delta_%",
        }
    )
    for col in ("full_%", "subset_%", "delta_%"):
        strata_table[col] = strata_table[col].map(lambda x: f"{x:.4f}")

    lines.append("--- Per stratum (set | spoof | label) ---")
    lines.append(strata_table.to_string(index=False))
    lines.append("")

    for title, cols in [
        ("Per split (set)", ["set"]),
        ("Per class (spoof)", ["spoof"]),
        (
            "Per deepfake label",
            ["label"],
        ),
    ]:
        if cols == ["label"]:
            full_part = full_df[full_df["spoof"] == "deepfake"]
            subset_part = subset_df[subset_df["spoof"] == "deepfake"]
        else:
            full_part = full_df
            subset_part = subset_df

        table = _distribution_table(full_part, subset_part, cols)
        for col in ("full_pct", "subset_pct", "delta_pct"):
            table[col] = table[col].map(lambda x: f"{x:.4f}")
        lines.append(f"--- {title} ---")
        lines.append(table.to_string(index=False))
        lines.append("")

    max_delta = 0.0
    for _, row in _distribution_table(stratum_full, stratum_subset, ["stratum_label"]).iterrows():
        max_delta = max(max_delta, abs(row["delta_pct"]))

    lines.append(f"Max |delta_%| across strata: {max_delta:.6f}")
    lines.append(f"Tolerance:                  {PROPORTION_TOLERANCE_PCT:.4f}")
    if max_delta <= PROPORTION_TOLERANCE_PCT:
        lines.append("PASS: subset proportions match full dataset within tolerance.")
    else:
        lines.append("FAIL: proportion deviation exceeds tolerance.")
    lines.append("=" * 72)

    return "\n".join(lines), max_delta


def copy_file(
    src: Path,
    dst: Path,
    skip_existing: bool,
) -> str:
    if skip_existing and dst.exists():
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def write_outputs(
    subset_df: pd.DataFrame,
    audio_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    fraction: float,
    seed: int,
    workers: int,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[int, int]:
    out_audio = output_dir / "all_data_16k_mono"
    out_meta_dir = output_dir / "all_data"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return 0, 0

    copied = 0
    skipped = 0
    tasks = []
    for _, row in subset_df.iterrows():
        src = audio_dir / row["filename"]
        dst = out_audio / row["filename"]
        tasks.append((src, dst))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(copy_file, src, dst, skip_existing): (src, dst)
            for src, dst in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying audio"):
            result = future.result()
            if result == "copied":
                copied += 1
            else:
                skipped += 1

    metadata_records: list[dict[str, Any]] = []
    for _, row in subset_df.iterrows():
        record = row.drop(labels=["stratum"]).to_dict()
        record["filepath"] = str(out_audio / row["filename"])
        metadata_records.append(record)

    metadata_path = out_meta_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=4, ensure_ascii=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "fraction": fraction,
        "source_metadata": str(metadata_path),
        "source_audio_dir": str(audio_dir),
        "output_dir": str(output_dir),
        "n_selected": len(subset_df),
        "files": [
            {
                "filename": row["filename"],
                "stratum": list(row["stratum"]),
            }
            for _, row in subset_df.iterrows()
        ],
    }
    manifest_path = out_meta_dir / "selection_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified benchmark subset of the SONICS dataset.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not (0 < args.fraction < 1):
        print("Error: --fraction must be in (0, 1).", file=sys.stderr)
        return 1

    t0 = time.perf_counter()

    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    df = pd.DataFrame(metadata)

    missing_mask = ~df["filename"].apply(lambda fn: (args.audio_dir / fn).exists())
    if missing_mask.any():
        n_missing = int(missing_mask.sum())
        print(f"Warning: {n_missing} metadata records have no audio file — excluding them.")
        df = df[~missing_mask].reset_index(drop=True)

    subset_df = select_subset(df, fraction=args.fraction, seed=args.seed)

    copied, skipped = write_outputs(
        subset_df=subset_df,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        fraction=args.fraction,
        seed=args.seed,
        workers=args.workers,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    elapsed = time.perf_counter() - t0
    report, max_delta = build_report(
        full_df=df,
        subset_df=subset_df,
        fraction=args.fraction,
        seed=args.seed,
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        elapsed_s=elapsed,
        copied=copied,
        skipped=skipped,
        dry_run=args.dry_run,
    )
    print(report)

    if not args.dry_run:
        report_path = args.output_dir / "selection_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

    if max_delta > PROPORTION_TOLERANCE_PCT:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

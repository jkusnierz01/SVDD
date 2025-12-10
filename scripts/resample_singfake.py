#!/usr/bin/env python3
"""
Batch resample audio files to 16kHz using ffmpeg.
Creates output directory next to input (default "<input_dir>_16k") and preserves filenames + subfolders.
Requires `ffmpeg` in PATH.
"""

from pathlib import Path
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import csv

def resample_one(in_path: Path, out_path: Path, sample_rate: int = 16000, mono: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return (str(in_path), str(out_path), "skipped", "")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(in_path)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate), str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        return (str(in_path), str(out_path), "ok", "")
    else:
        return (str(in_path), str(out_path), "error", proc.stderr.strip())

def gather_files(input_dir: Path, exts):
    return [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def main():
    parser = argparse.ArgumentParser(description="Resample audio folder to 16kHz (ffmpeg)")
    parser.add_argument("--input-dir", "-i", required=True, type=Path, help="Input folder with audio files")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="Output folder (default: sibling named <input>_16k)")
    parser.add_argument("--ext", "-e", nargs="+", default=[".flac", ".wav", ".mp3", ".m4a", ".ogg"], help="Extensions to process")
    parser.add_argument("--workers", "-w", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Number of parallel ffmpeg jobs")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--mono", action="store_true", help="Force mono output (default: True)", default=True)
    parser.add_argument("--no-mono", dest="mono", action="store_false", help="Do not force mono")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files (default: skip)")

    args = parser.parse_args()
    in_dir = args.input_dir.resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input dir not found: {in_dir}")

    out_dir = args.out_dir.resolve() if args.out_dir else in_dir.parent / f"{in_dir.name}_16k"
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}

    files = gather_files(in_dir, exts)
    if not files:
        print("No files found for extensions:", exts)
        return

    print(f"Found {len(files)} files — output -> {out_dir}, workers={args.workers}, sr={args.sr}, mono={args.mono}")

    tasks = []
    results = []
    failed = []

    def submit_jobs(executor):
        futures = {}
        for p in files:
            rel = p.relative_to(in_dir)
            out_path = out_dir / rel
            if not args.overwrite and out_path.exists():
                results.append((str(p), str(out_path), "skipped", "exists"))
                continue
            futures[executor.submit(resample_one, p, out_path, args.sr, args.mono)] = p
        return futures

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = submit_jobs(ex)
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            results.append(res)
            if res[2] != "ok" and res[2] != "skipped":
                failed.append(res)

    # summary
    ok = sum(1 for r in results if r[2] == "ok")
    skipped = sum(1 for r in results if r[2] == "skipped")
    errors = len(failed)
    print(f"Done. ok={ok}, skipped={skipped}, errors={errors}")

    # write CSV of errors
    csv_out = out_dir / "resample_results.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["input", "output", "status", "message"])
        for r in results:
            writer.writerow(r)

    if failed:
        print("Failed examples (first 10):")
        for row in failed[:10]:
            print(row[0], "->", row[3][:200])

if __name__ == "__main__":
    main()
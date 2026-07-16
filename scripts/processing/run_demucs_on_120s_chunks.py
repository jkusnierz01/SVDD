#!/usr/bin/env python3
"""Run Demucs (mdx_extra, two-stems) on pre-cut 120s benchmark chunks.

Output layout (matches demucs CLI --two-stems=vocals):
  {output_dir}/{model}/{track_stem}/vocals.wav
  {output_dir}/{model}/{track_stem}/no_vocals.wav
"""

import argparse
import gc
import json
import subprocess
import sys
import time
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio, prevent_clip
from demucs.pretrained import get_model
from tqdm import tqdm

SONICS_BASE = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics")
DEFAULT_INPUT = SONICS_BASE / "benchmark_10pct" / "chunks_120s"
DEFAULT_OUTPUT = SONICS_BASE / "benchmark_10pct" / "demucs_120s"

DEMUCS_MODEL = "mdx_extra"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Demucs separation on 120s chunks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEMUCS_MODEL)
    parser.add_argument(
        "--backend",
        type=str,
        default="python",
        choices=["python", "subprocess"],
        help="python: single-process apply_model (fast). subprocess: demucs CLI fallback.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=1, help="Demucs CLI -j (subprocess backend only)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Subprocess backend: files per demucs call (0 = entire shard in one call)",
    )
    parser.add_argument("--segment", type=int, default=None, help="Override Demucs segment length (seconds)")
    parser.add_argument("--shifts", type=int, default=0, help="Demucs time-shift averaging (0 = faster)")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard index for SLURM array")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shard_files(files: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    if num_shards <= 1:
        return files
    return [f for i, f in enumerate(files) if i % num_shards == shard_id]


def track_dir(output_dir: Path, model: str, stem: str) -> Path:
    return output_dir / model / stem


def vocals_path(output_dir: Path, model: str, stem: str) -> Path:
    return track_dir(output_dir, model, stem) / "vocals.wav"


def no_vocals_path(output_dir: Path, model: str, stem: str) -> Path:
    return track_dir(output_dir, model, stem) / "no_vocals.wav"


def stems_complete(output_dir: Path, model: str, stem: str) -> bool:
    return vocals_path(output_dir, model, stem).exists() and no_vocals_path(output_dir, model, stem).exists()


def load_demucs_model(model_name: str, device: torch.device):
    model = get_model(model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_chunk_flac(path: Path, channels: int, samplerate: int) -> torch.Tensor:
    """Load 16 kHz mono FLAC via soundfile (fast on PLGrid scratch)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return convert_audio(wav, sr, samplerate, channels)


def save_stem_wav(wav: torch.Tensor, path: Path, samplerate: int) -> None:
    """Save stem via soundfile (avoids slow torchcodec/torchaudio)."""
    wav = prevent_clip(wav, mode="rescale")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.cpu().numpy().T, samplerate, subtype="PCM_16")


def separate_stems_python(
    track_path: Path,
    model,
    device: torch.device,
    out_dir: Path,
    segment: int | None,
    shifts: int,
    fp16: bool,
) -> None:
    wav = load_chunk_flac(track_path, model.audio_channels, model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    mix = wav[None].to(device)

    with torch.no_grad():
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if fp16 and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            sources = apply_model(
                model,
                mix,
                device=device,
                shifts=shifts,
                split=True,
                overlap=0.25,
                progress=False,
                segment=segment,
            )

    sources = sources[0]
    sources = sources * ref.std().to(sources.device) + ref.mean().to(sources.device)

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]
    no_vocals = torch.zeros_like(vocals)
    for i, name in enumerate(model.sources):
        if name != "vocals":
            no_vocals += sources[i]

    sr = model.samplerate
    save_stem_wav(vocals, out_dir / "vocals.wav", sr)
    save_stem_wav(no_vocals, out_dir / "no_vocals.wav", sr)


def run_demucs_subprocess_batch(
    files: list[Path],
    output_dir: Path,
    model: str,
    device: str,
    jobs: int,
    segment: int | None,
    shifts: int,
) -> None:
    cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n", model,
        "-d", device,
        "-j", str(jobs),
        "--shifts", str(shifts),
        "-o", str(output_dir),
    ]
    if segment is not None:
        cmd.extend(["--segment", str(segment)])
    cmd.extend(str(f) for f in files)
    subprocess.run(cmd, check=True)


def run_python_backend(
    pending: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, int]:
    model = load_demucs_model(args.model, device)
    ok = errors = 0

    for idx, track_path in enumerate(tqdm(pending, desc=f"Demucs shard {args.shard_id}")):
        out_dir = track_dir(args.output_dir, args.model, track_path.stem)
        try:
            separate_stems_python(
                track_path, model, device, out_dir, args.segment, args.shifts, args.fp16
            )
            ok += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR {track_path.stem}: {exc}", file=sys.stderr)
        if device.type == "cuda" and (idx + 1) % 32 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return ok, errors


def run_subprocess_backend(
    pending: list[Path],
    args: argparse.Namespace,
) -> tuple[int, int]:
    batch_size = args.batch_size if args.batch_size > 0 else len(pending)
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    ok = errors = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Demucs shard {args.shard_id}")):
        try:
            run_demucs_subprocess_batch(
                batch, args.output_dir, args.model, args.device, args.jobs,
                args.segment, args.shifts,
            )
            ok += len(batch)
        except subprocess.CalledProcessError as exc:
            errors += len(batch)
            print(f"Batch {batch_idx + 1}/{len(batches)} failed: {exc}", file=sys.stderr)

    return ok, errors


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", message=".*TorchCodec.*")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.", file=sys.stderr)
        args.device = "cpu"
        args.fp16 = False

    all_files = sorted(args.input_dir.glob("*.flac"))
    if args.limit is not None:
        all_files = all_files[: args.limit]

    all_files = shard_files(all_files, args.shard_id, args.num_shards)
    if not all_files:
        print("No files to process in this shard.")
        return 0

    pending: list[Path] = []
    for f in all_files:
        if args.skip_existing and stems_complete(args.output_dir, args.model, f.stem):
            continue
        pending.append(f)

    skipped = len(all_files) - len(pending)
    print(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(all_files)} assigned, {len(pending)} to run, {skipped} skipped | "
        f"backend={args.backend} fp16={args.fp16} shifts={args.shifts}"
    )

    if args.dry_run:
        for f in pending[:5]:
            td = track_dir(args.output_dir, args.model, f.stem)
            print(f"Would process {f.name} -> {td}/vocals.wav + {td}/no_vocals.wav")
        return 0

    if not pending:
        print("All files already separated.")
        return 0

    t0 = time.perf_counter()
    device = torch.device(args.device)

    if args.backend == "python":
        ok, errors = run_python_backend(pending, args, device)
    else:
        ok, errors = run_subprocess_backend(pending, args)

    elapsed = time.perf_counter() - t0
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "backend": args.backend,
        "model": args.model,
        "device": args.device,
        "fp16": args.fp16,
        "jobs": args.jobs,
        "batch_size": args.batch_size,
        "segment": args.segment,
        "shifts": args.shifts,
        "assigned": len(all_files),
        "skipped": skipped,
        "processed_ok": ok,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "throughput_files_per_hour": round(ok / (elapsed / 3600), 2) if elapsed > 0 else 0,
    }
    report_path = args.output_dir / f"demucs_report_shard{args.shard_id:03d}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

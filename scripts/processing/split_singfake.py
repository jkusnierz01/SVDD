import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


"""
---------------------------------------------------
SCRIPT GETS BOTH VOCALS AND MIXTURES AND SPLIT THEM WITH VAD TIMESTAMPS

of course you can process vocals the same way
---------------------------------------------------
"""


DEFAULT_OUTPUT_PATH = Path(
    "/mnt/data/kusnierz/audio-data/SingFake/data_after_separation"
)
LABEL_MAP = {"bonafide": 1, "spoof": 0}

_VAD_LINE_RE = re.compile(
    r"\[\s*(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\]"
)


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float


def _parse_timestamp(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {ts!r}")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return seconds + 60.0 * minutes + 3600.0 * hours


def parse_vad_file(vad_path: Path) -> list[Segment]:
    segments: list[Segment] = []
    with open(vad_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = _VAD_LINE_RE.search(line)
            if not match:
                continue
            start_s = _parse_timestamp(match.group("start"))
            end_s = _parse_timestamp(match.group("end"))
            if end_s <= start_s:
                continue
            segments.append(Segment(start_s=start_s, end_s=end_s))
    return segments


def infer_label_from_stem(stem: str) -> int:
    # bonafide/spoof mapped to int
    parts = stem.split("_")
    if not parts:
        return -1
    label_str = parts[-1].lower()
    return LABEL_MAP.get(label_str, -1)


def find_mixture_file(download_dump_dir: Path, stem: str) -> Path | None:
    exts = [".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus"]
    for ext in exts:
        candidate = download_dump_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: slow scan
    for p in download_dump_dir.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def to_frames_channels(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.expand_dims(x, axis=1)
    if x.ndim == 2:
        # librosa.load(mono=False) returns (channels, frames)
        return np.transpose(x)
    raise ValueError(f"Unexpected audio shape: {x.shape}")


def slice_audio(x: np.ndarray, start_s: float, end_s: float, sr: int) -> np.ndarray:
    start = int(round(start_s * sr))
    end = int(round(end_s * sr))
    if x.ndim == 1:
        return x[max(start, 0) : max(end, 0)]
    return x[:, max(start, 0) : max(end, 0)]


def vad_single(
    vad_path, download_dump_dir, output_sr, vocals_out_dir, mixtures_out_dir
):
    vocals_path = vad_path.with_suffix(".wav")
    if not vocals_path.exists():
        print(f"No vocals: {vocals_path}, skip.")
        return

    track_stem = vad_path.parent.name
    label_int = infer_label_from_stem(track_stem)
    if label_int == -1:
        print(
            f"Unknown label: {track_stem} skipping..."
        )
        return

    mixture_path = find_mixture_file(download_dump_dir, track_stem)
    if mixture_path is None:
        print(
            f"No mixture for: {track_stem} w {download_dump_dir}, skipping"
        )
        return

    segments = parse_vad_file(vad_path)
    if not segments:
        return

    try:
        vocals, _ = librosa.load(str(vocals_path), sr=output_sr, mono=False)
        mixture, _ = librosa.load(str(mixture_path), sr=output_sr, mono=False)
    except Exception as e:
        print(f"Error loading audio ({track_stem}): {e}")
        return

    index = 0
    for seg in segments:
        try:
            vocals_seg = slice_audio(vocals, seg.start_s, seg.end_s, output_sr)
            mixture_seg = slice_audio(mixture, seg.start_s, seg.end_s, output_sr)

            # skip empty slices
            if vocals_seg.size == 0 or mixture_seg.size == 0:
                continue

            vocals_seg = to_frames_channels(vocals_seg)
            mixture_seg = to_frames_channels(mixture_seg)

            ## Here we specify .FLAC files!
            file_name = f"{track_stem}__seg{index:05d}.flac"
            vocals_target = vocals_out_dir / file_name
            mixture_target = mixtures_out_dir / file_name

            if vocals_target.exists() and mixture_target.exists():
                index += 1
                continue

            sf.write(
                str(vocals_target),
                vocals_seg,
                output_sr,
                subtype="PCM_16",
                format="FLAC",
            )
            sf.write(
                str(mixture_target),
                mixture_seg,
                output_sr,
                subtype="PCM_16",
                format="FLAC",
            )
            index += 1
        except Exception as e:
            print(f"Błąd zapisu segmentu ({track_stem}): {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Split SingFake into short segments using .vad from separate_singfake.py"
    )
    parser.add_argument(
        "--download_dump_dir",
        required=True,
        help="directory with the original downloaded mixtures (e.g., .flac files)",
    )
    parser.add_argument(
        "--separation_output_dir",
        default=str(DEFAULT_OUTPUT_PATH),
        help="base output dir used by separate_singfake.py (contains mdx_extra/*/vocals.wav + .vad)",
    )
    parser.add_argument(
        "--dump_dir",
        default=None,
        help="output directory for the split dataset (will contain vocals/ and mixtures/). Default: <separation_output_dir>/split_dump",
    )
    parser.add_argument(
        "--output_sr",
        type=int,
        default=16000,
        help="sample rate for output segments",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="number of worker processes (default: min(cpu_count, 8))",
    )
    args = parser.parse_args()

    download_dump_dir = Path(args.download_dump_dir)
    separation_output_dir = Path(args.separation_output_dir)
    dump_dir = (
        Path(args.dump_dir) if args.dump_dir else (separation_output_dir / "split_dump")
    )
    output_sr: int = args.output_sr

    mdx_dir = separation_output_dir / "mdx_extra"
    assert mdx_dir.exists(), f"Missing mdx_extra directory: {mdx_dir}"

    vocals_out_dir = dump_dir / "vocals"
    mixtures_out_dir = dump_dir / "mixtures"
    vocals_out_dir.mkdir(parents=True, exist_ok=True)
    mixtures_out_dir.mkdir(parents=True, exist_ok=True)

    vad_files: list[Path] = []
    for root, _, files in os.walk(mdx_dir):
        for file in files:
            if file.endswith(".vad"):
                vad_files.append(Path(root) / file)

    if not vad_files:
        print(f"Nie znaleziono plików .vad pod: {mdx_dir}")
        return

    default_workers = min((os.cpu_count() or 1), 8)
    max_workers = (
        args.max_workers
        if (args.max_workers and args.max_workers > 0)
        else default_workers
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                vad_single,
                vad_path,
                download_dump_dir,
                output_sr,
                vocals_out_dir,
                mixtures_out_dir,
            )
            for vad_path in vad_files
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Rendering splits"
        ):
            try:
                fut.result()
            except Exception as e:
                print(f"Worker error: {e}")


if __name__ == "__main__":
    main()

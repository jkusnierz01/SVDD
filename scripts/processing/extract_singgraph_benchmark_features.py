#!/usr/bin/env python3
"""Extract SingGraph fp16 features from pre-separated Demucs stems (benchmark subset).

Flow per track:
  vocals.wav + no_vocals.wav -> mono 16k -> RawBoost (vocals) -> 30x4s windows
  -> W2V XLS-R (vocals) + MERT 330M (instrumental) -> [30, T, 1024] fp16 .pt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "processing"))

from src.preprocessing.feature_align import align_time_axis_linear

from benchmark_feature_common import (
    DEFAULT_DEMUCS_DIR,
    DEFAULT_OUTPUT,
    DEMUCS_MODEL,
    FEAT_DIM,
    MERT_330M,
    MERT_SR,
    SINGGRAPH_NUM_WINDOWS,
    SINGGRAPH_WINDOW_SAMPLES,
    SR,
    W2V_XLSR,
    SingGraphPaths,
    apply_rawboost_vocals,
    demucs_instrumental_path,
    demucs_vocals_path,
    load_demucs_stem,
    load_index,
    save_fp16,
    merge_index_entry,
    save_index,
    shard_files,
    windowize_4s,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_train_stem(stem: str) -> bool:
    return stem.startswith("train_")


def _should_rawboost(stem: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "train":
        return _is_train_stem(stem)
    return False


@dataclass
class SingGraphBundle:
    device: torch.device
    resampler_mert: T.Resample = field(init=False)

    w2v_processor: Wav2Vec2FeatureExtractor = field(init=False)
    w2v_model: Wav2Vec2Model = field(init=False)
    mert_processor: Wav2Vec2FeatureExtractor = field(init=False)
    mert_model: AutoModel = field(init=False)

    def __post_init__(self) -> None:
        dev = str(self.device)
        self.resampler_mert = T.Resample(orig_freq=SR, new_freq=MERT_SR)

        logger.info("Loading W2V XLS-R (%s)", W2V_XLSR)
        self.w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_XLSR)
        self.w2v_model = Wav2Vec2Model.from_pretrained(W2V_XLSR).to(dev)

        logger.info("Loading MERT 330M (%s)", MERT_330M)
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_330M, trust_remote_code=True
        )
        self.mert_model = AutoModel.from_pretrained(MERT_330M, trust_remote_code=True).to(dev)

        for model in (self.w2v_model, self.mert_model):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def _w2v_last(self, batch_wav: torch.Tensor) -> torch.Tensor:
        """[B, 64000] -> [B, T, 1024] on CPU float32."""
        inputs = self.w2v_processor(
            batch_wav.numpy(),
            sampling_rate=SR,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                out = self.w2v_model(inputs.input_values.to(self.device)).last_hidden_state
        return out.float().cpu()

    def _mert_last(self, batch_wav_16k: torch.Tensor) -> torch.Tensor:
        """[B, 64000] -> [B, T, 1024] on CPU float32."""
        batch_24k = self.resampler_mert(batch_wav_16k)
        inputs = self.mert_processor(
            batch_24k.numpy(),
            sampling_rate=MERT_SR,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                out = self.mert_model(inputs.input_values.to(self.device)).last_hidden_state
        return out.float().cpu()

    def extract(
        self,
        vocals_wav: torch.Tensor,
        inst_wav: torch.Tensor,
        stem: str,
        paths: SingGraphPaths,
        rawboost_mode: str,
        apply_rawboost: bool,
    ) -> None:
        vocals = apply_rawboost_vocals(vocals_wav, stem, mode=rawboost_mode) if apply_rawboost else vocals_wav
        v_windows = windowize_4s(vocals)
        i_windows = windowize_4s(inst_wav)

        if v_windows.shape[0] != SINGGRAPH_NUM_WINDOWS:
            raise ValueError(
                f"Expected {SINGGRAPH_NUM_WINDOWS} vocal windows, got {v_windows.shape[0]}"
            )
        if i_windows.shape[0] != SINGGRAPH_NUM_WINDOWS:
            raise ValueError(
                f"Expected {SINGGRAPH_NUM_WINDOWS} instrumental windows, got {i_windows.shape[0]}"
            )
        if v_windows.shape[1] != SINGGRAPH_WINDOW_SAMPLES:
            raise ValueError(f"Unexpected window size: {v_windows.shape[1]}")

        v_feat = self._w2v_last(v_windows)
        i_feat = self._mert_last(i_windows)

        if v_feat.shape[0] != SINGGRAPH_NUM_WINDOWS or v_feat.shape[2] != FEAT_DIM:
            raise ValueError(f"Unexpected vocal feature shape: {tuple(v_feat.shape)}")
        if i_feat.shape[0] != SINGGRAPH_NUM_WINDOWS or i_feat.shape[2] != FEAT_DIM:
            raise ValueError(f"Unexpected instrumental feature shape: {tuple(i_feat.shape)}")

        i_feat = align_time_axis_linear(i_feat, target_len=v_feat.shape[1])
        if i_feat.shape != v_feat.shape:
            raise ValueError(
                f"SingGraph align failed for {stem}: "
                f"vocals={tuple(v_feat.shape)} instrumental={tuple(i_feat.shape)}"
            )

        save_fp16(v_feat, paths.vocals)
        save_fp16(i_feat, paths.instrumental)


def collect_stems(demucs_dir: Path) -> list[str]:
    model_dir = demucs_dir / DEMUCS_MODEL
    if not model_dir.is_dir():
        return []
    stems = []
    for track_dir in sorted(model_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        if demucs_vocals_path(demucs_dir, track_dir.name).exists() and demucs_instrumental_path(
            demucs_dir, track_dir.name
        ).exists():
            stems.append(track_dir.name)
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SingGraph features from Demucs stems.")
    parser.add_argument("--demucs-dir", type=Path, default=DEFAULT_DEMUCS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rawboost-mode", type=str, default="ssi", choices=["ssi", "full", "none"])
    parser.add_argument(
        "--rawboost-scope",
        type=str,
        default="train",
        choices=["all", "train", "none"],
        help="Where to apply RawBoost on vocal stems.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stems = collect_stems(args.demucs_dir)
    if args.limit is not None:
        all_stems = all_stems[: args.limit]
    all_stems = shard_files([Path(s) for s in all_stems], args.shard_id, args.num_shards)
    all_stems = [p.name if isinstance(p, Path) else str(p) for p in all_stems]

    if not all_stems:
        logger.info("No Demucs tracks in this shard.")
        return 0

    if args.dry_run:
        for stem in all_stems[:5]:
            paths = SingGraphPaths.for_stem(args.output_dir, stem)
            logger.info(
                "Would process %s -> %s, %s",
                stem,
                paths.vocals,
                paths.instrumental,
            )
        logger.info("Dry run: %d tracks in shard %d/%d", len(all_stems), args.shard_id, args.num_shards)
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bundle = SingGraphBundle(device=device)

    index_path = args.output_dir / "index.json"
    index = load_index(index_path)

    errors_log = args.output_dir / f"singgraph_errors_shard{args.shard_id:03d}.log"
    ok = skipped = errors = 0
    sample_shapes: dict[str, tuple] | None = None
    t0 = time.perf_counter()

    logger.info(
        "Shard %d/%d: %d tracks | rawboost=%s rawboost_scope=%s",
        args.shard_id,
        args.num_shards,
        len(all_stems),
        args.rawboost_mode,
        args.rawboost_scope,
    )
    rawboosted_items = 0

    for stem in tqdm(all_stems, desc=f"SingGraph shard {args.shard_id}"):
        paths = SingGraphPaths.for_stem(args.output_dir, stem)
        if args.skip_existing and paths.complete():
            skipped += 1
            continue

        vocals_path = demucs_vocals_path(args.demucs_dir, stem)
        inst_path = demucs_instrumental_path(args.demucs_dir, stem)

        try:
            with torch.no_grad():
                vocals = load_demucs_stem(vocals_path)
                inst = load_demucs_stem(inst_path)
                use_rawboost = args.rawboost_mode != "none" and _should_rawboost(stem, args.rawboost_scope)
                if use_rawboost:
                    rawboosted_items += 1
                bundle.extract(
                    vocals,
                    inst,
                    stem,
                    paths,
                    args.rawboost_mode if args.rawboost_mode != "none" else "ssi",
                    apply_rawboost=use_rawboost,
                )

            if sample_shapes is None:
                sample_shapes = {
                    "singgraph_vocals": tuple(torch.load(paths.vocals, weights_only=True).shape),
                    "singgraph_instrumental": tuple(
                        torch.load(paths.instrumental, weights_only=True).shape
                    ),
                }

            merge_index_entry(index, paths.as_index_entry(stem))
            ok += 1
        except Exception as exc:
            errors += 1
            msg = f"{stem}: {exc}\n{traceback.format_exc()}\n"
            with open(errors_log, "a", encoding="utf-8") as f:
                f.write(msg)
            logger.error("Failed %s: %s", stem, exc)
        finally:
            cleanup_gpu()

    elapsed = time.perf_counter() - t0
    peak_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    save_index(index_path, index)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "demucs_dir": str(args.demucs_dir),
        "output_dir": str(args.output_dir),
        "rawboost_mode": args.rawboost_mode,
        "rawboost_scope": args.rawboost_scope,
        "rawboosted_items": rawboosted_items,
        "assigned": len(all_stems),
        "processed_ok": ok,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "throughput_files_per_hour": round(ok / (elapsed / 3600), 2) if elapsed > 0 else 0,
        "peak_vram_gb": round(peak_vram, 2) if peak_vram is not None else None,
        "sample_shapes": sample_shapes,
    }
    report_path = args.output_dir / f"singgraph_report_shard{args.shard_id:03d}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

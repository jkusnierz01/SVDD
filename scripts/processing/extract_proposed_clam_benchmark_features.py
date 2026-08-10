#!/usr/bin/env python3
"""Extract Proposed + CLAM fp16 features from pre-cut 120s benchmark chunks.

Flow per file:
  read 120s FLAC -> augment once -> Proposed (full 120s, pool [1499,1024])
                                 -> CLAM (first 90s of aug, all layers [T,13,768])
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

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "processing"))

from src.preprocessing.clam_align import align_clam_time_axis

from benchmark_feature_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    FEAT_DIM,
    MERT_330M,
    MERT_95M,
    MERT_SR,
    PROPOSED_CHUNK_SEC,
    PROPOSED_POOLED_LEN,
    PROPOSED_RAW_SEQ_LEN,
    PROPOSED_SEC,
    SR,
    W2V_CLAM,
    W2V_XLSR,
    OutputPaths,
    augment_chunk,
    clam_clip_90s,
    load_index,
    merge_index_entry,
    pool_proposed,
    read_chunk_flac,
    save_fp16,
    save_index,
    shard_files,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _is_train_stem(stem: str) -> bool:
    return stem.startswith("train_")


def _should_augment(stem: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "train":
        return _is_train_stem(stem)
    return False


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class ProposedClamBundle:
    device: torch.device
    resampler_mert: T.Resample = field(init=False)

    w2v_xlsr_processor: Wav2Vec2FeatureExtractor = field(init=False)
    w2v_xlsr_model: Wav2Vec2Model = field(init=False)
    mert_330m_processor: Wav2Vec2FeatureExtractor = field(init=False)
    mert_330m_model: AutoModel = field(init=False)

    w2v_clam_processor: Wav2Vec2FeatureExtractor = field(init=False)
    w2v_clam_model: AutoModel = field(init=False)
    mert_95m_processor: Wav2Vec2FeatureExtractor = field(init=False)
    mert_95m_model: AutoModel = field(init=False)

    def __post_init__(self) -> None:
        dev = str(self.device)
        self.resampler_mert = T.Resample(orig_freq=SR, new_freq=MERT_SR)

        logger.info("Loading W2V XLS-R (%s)", W2V_XLSR)
        self.w2v_xlsr_processor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_XLSR)
        self.w2v_xlsr_model = Wav2Vec2Model.from_pretrained(W2V_XLSR).to(dev)

        logger.info("Loading MERT 330M (%s)", MERT_330M)
        self.mert_330m_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_330M, trust_remote_code=True
        )
        self.mert_330m_model = AutoModel.from_pretrained(MERT_330M, trust_remote_code=True).to(dev)

        logger.info("Loading CLAM W2V (%s)", W2V_CLAM)
        self.w2v_clam_processor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_CLAM)
        self.w2v_clam_model = AutoModel.from_pretrained(W2V_CLAM, trust_remote_code=True).to(dev)

        logger.info("Loading CLAM MERT (%s)", MERT_95M)
        self.mert_95m_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_95M, trust_remote_code=True
        )
        self.mert_95m_model = AutoModel.from_pretrained(MERT_95M, trust_remote_code=True).to(dev)

        for model in (
            self.w2v_xlsr_model,
            self.mert_330m_model,
            self.w2v_clam_model,
            self.mert_95m_model,
        ):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def _w2v_xlsr_last(self, batch_wav: torch.Tensor) -> torch.Tensor:
        inputs = self.w2v_xlsr_processor(
            batch_wav.numpy(),
            sampling_rate=SR,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                out = self.w2v_xlsr_model(inputs.input_values.to(self.device)).last_hidden_state
        return out.float().cpu()

    def _mert_330m_last(self, batch_wav_16k: torch.Tensor) -> torch.Tensor:
        batch_24k = self.resampler_mert(batch_wav_16k)
        inputs = self.mert_330m_processor(
            batch_24k.numpy(),
            sampling_rate=MERT_SR,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                out = self.mert_330m_model(inputs.input_values.to(self.device)).last_hidden_state
        return out.float().cpu()

    def _clam_all_layers(
        self, model: AutoModel, processor: Wav2Vec2FeatureExtractor, wav_16k: torch.Tensor, mert: bool
    ) -> torch.Tensor:
        if mert:
            wav = self.resampler_mert(wav_16k)
            sr = MERT_SR
        else:
            wav = wav_16k
            sr = SR

        inputs = processor(
            wav.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                out = model(
                    **{k: v.to(self.device) for k, v in inputs.items()},
                    output_hidden_states=True,
                )
        layers = torch.stack(out.hidden_states, dim=1).squeeze(0)
        layers = layers.permute(1, 0, 2).contiguous()
        if layers.shape[1] != 13 or layers.shape[2] != 768:
            raise ValueError(f"Unexpected CLAM layer shape: {tuple(layers.shape)}")
        return layers.float().cpu()

    def extract_proposed(self, aug_wav: torch.Tensor, paths: OutputPaths) -> None:
        chunk_samples = PROPOSED_CHUNK_SEC * SR
        num_chunks = PROPOSED_SEC // PROPOSED_CHUNK_SEC
        chunks = aug_wav.squeeze(0).unfold(0, chunk_samples, chunk_samples)

        out_w2v = self._w2v_xlsr_last(chunks)
        out_mert = self._mert_330m_last(chunks)

        out_mert = F.interpolate(
            out_mert.transpose(1, 2),
            size=out_w2v.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        final_w2v = out_w2v.reshape(num_chunks * out_w2v.shape[1], out_w2v.shape[2])
        final_mert = out_mert.reshape(num_chunks * out_mert.shape[1], out_mert.shape[2])

        if final_w2v.shape[0] != PROPOSED_RAW_SEQ_LEN:
            logger.warning(
                "Proposed W2V raw seq %s (expected %d)", tuple(final_w2v.shape), PROPOSED_RAW_SEQ_LEN
            )
        if final_mert.shape[0] != PROPOSED_RAW_SEQ_LEN:
            logger.warning(
                "Proposed MERT raw seq %s (expected %d)", tuple(final_mert.shape), PROPOSED_RAW_SEQ_LEN
            )

        pooled_w2v = pool_proposed(final_w2v)
        pooled_mert = pool_proposed(final_mert)

        if pooled_w2v.shape != (PROPOSED_POOLED_LEN, FEAT_DIM):
            raise ValueError(
                f"Proposed pooled W2V shape {tuple(pooled_w2v.shape)}, "
                f"expected ({PROPOSED_POOLED_LEN}, {FEAT_DIM})"
            )
        if pooled_mert.shape != (PROPOSED_POOLED_LEN, FEAT_DIM):
            raise ValueError(
                f"Proposed pooled MERT shape {tuple(pooled_mert.shape)}, "
                f"expected ({PROPOSED_POOLED_LEN}, {FEAT_DIM})"
            )

        save_fp16(pooled_w2v, paths.proposed_w2v)
        save_fp16(pooled_mert, paths.proposed_mert)

        pooled = torch.cat([pooled_w2v, pooled_mert], dim=1)
        paths.proposed_pooled.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(paths.proposed_pooled), pooled.half().numpy())

    def extract_clam(self, aug_wav: torch.Tensor, paths: OutputPaths) -> None:
        clam_wav = clam_clip_90s(aug_wav)
        w2v_layers = self._clam_all_layers(
            self.w2v_clam_model, self.w2v_clam_processor, clam_wav, mert=False
        )
        mert_layers = self._clam_all_layers(
            self.mert_95m_model, self.mert_95m_processor, clam_wav, mert=True
        )
        mert_layers = align_clam_time_axis(mert_layers, target_len=w2v_layers.shape[0])
        if mert_layers.shape != w2v_layers.shape:
            raise ValueError(
                f"CLAM align failed for {paths.clam_w2v.stem}: "
                f"w2v={tuple(w2v_layers.shape)} mert={tuple(mert_layers.shape)}"
            )
        save_fp16(w2v_layers, paths.clam_w2v)
        save_fp16(mert_layers, paths.clam_mert)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Proposed + CLAM features from 120s benchmark chunks."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aug-variant", type=int, default=0)
    parser.add_argument(
        "--augment-scope",
        type=str,
        default="train",
        choices=["all", "train", "none"],
        help="Where to apply offline audio augmentation.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-proposed", action="store_true")
    parser.add_argument("--skip-clam", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(args.input_dir.glob("*.flac"))
    if args.limit is not None:
        all_files = all_files[: args.limit]
    all_files = shard_files(all_files, args.shard_id, args.num_shards)

    if not all_files:
        logger.info("No files in this shard.")
        return 0

    if args.dry_run:
        for flac in all_files[:5]:
            paths = OutputPaths.for_stem(args.output_dir, flac.stem)
            logger.info(
                "Would process %s -> proposed %s, clam %s",
                flac.name,
                paths.proposed_w2v.parent,
                paths.clam_w2v.parent,
            )
        logger.info("Dry run: %d files in shard %d/%d", len(all_files), args.shard_id, args.num_shards)
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bundle = ProposedClamBundle(device=device)

    index_path = args.output_dir / "index.json"
    index = load_index(index_path)

    errors_log = args.output_dir / f"errors_shard{args.shard_id:03d}.log"
    ok = skipped = errors = 0
    sample_shapes: dict[str, tuple] | None = None
    t0 = time.perf_counter()

    logger.info(
        "Shard %d/%d: %d files | skip_proposed=%s skip_clam=%s aug_variant=%d augment_scope=%s",
        args.shard_id,
        args.num_shards,
        len(all_files),
        args.skip_proposed,
        args.skip_clam,
        args.aug_variant,
        args.augment_scope,
    )
    augmented_items = 0

    for flac in tqdm(all_files, desc=f"Features shard {args.shard_id}"):
        stem = flac.stem
        paths = OutputPaths.for_stem(args.output_dir, stem)

        if args.skip_existing and paths.should_skip(
            skip_proposed=args.skip_proposed, skip_clam=args.skip_clam
        ):
            merge_index_entry(index, paths.as_index_entry(stem))
            skipped += 1
            continue

        try:
            wav = read_chunk_flac(flac)
            if _should_augment(stem, args.augment_scope):
                aug_wav = augment_chunk(wav, stem, args.aug_variant)
                augmented_items += 1
            else:
                aug_wav = wav

            if not args.skip_proposed and not (args.skip_existing and paths.proposed_complete()):
                bundle.extract_proposed(aug_wav, paths)

            if not args.skip_clam and not (args.skip_existing and paths.clam_complete()):
                bundle.extract_clam(aug_wav, paths)

            if sample_shapes is None:
                sample_shapes = {}
                if paths.proposed_w2v.exists():
                    sample_shapes["proposed_w2v"] = tuple(
                        torch.load(paths.proposed_w2v, weights_only=True).shape
                    )
                if paths.proposed_mert.exists():
                    sample_shapes["proposed_mert"] = tuple(
                        torch.load(paths.proposed_mert, weights_only=True).shape
                    )
                if paths.clam_w2v.exists():
                    sample_shapes["clam_w2v"] = tuple(torch.load(paths.clam_w2v, weights_only=True).shape)
                if paths.clam_mert.exists():
                    sample_shapes["clam_mert"] = tuple(
                        torch.load(paths.clam_mert, weights_only=True).shape
                    )

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
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "aug_variant": args.aug_variant,
        "augment_scope": args.augment_scope,
        "augmented_items": augmented_items,
        "skip_proposed": args.skip_proposed,
        "skip_clam": args.skip_clam,
        "assigned": len(all_files),
        "processed_ok": ok,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "throughput_files_per_hour": round(ok / (elapsed / 3600), 2) if elapsed > 0 else 0,
        "peak_vram_gb": round(peak_vram, 2) if peak_vram is not None else None,
        "sample_shapes": sample_shapes,
    }
    report_path = args.output_dir / f"features_report_shard{args.shard_id:03d}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

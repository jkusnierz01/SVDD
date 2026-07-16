#!/usr/bin/env python3
"""Offline fp16 feature extraction for SONICS benchmark subset (legacy monolith).

Note: Proposed + CLAM branches from 120s chunks are handled by
``extract_proposed_clam_benchmark_features.py``. SingGraph from pre-separated
Demucs stems is handled by ``extract_singgraph_benchmark_features.py``. This
script remains for legacy all-in-one extraction from raw audio.
"""

import argparse
import gc
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_augmentation import apply_audio_augment, aug_seed
from src.utils.preprocessing import pad_loop_torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SONICS_BASE = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics")
DEFAULT_INPUT = SONICS_BASE / "benchmark_10pct" / "all_data_16k_mono"
DEFAULT_OUTPUT = SONICS_BASE / "benchmark_10pct" / "features"

SR = 16_000
MERT_SR = 24_000
PROPOSED_SEC = 120
PROPOSED_CHUNK_SEC = 30
CLAM_SEC = 90
SINGGRAPH_WINDOW_SEC = 4

W2V_XLSR = "facebook/wav2vec2-xls-r-300m"
MERT_330M = "m-a-p/MERT-v1-330M"
W2V_CLAM = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
MERT_95M = "m-a-p/MERT-v1-95M"
DEMUCS_MODEL = "mdx_extra"


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_fp16(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().half(), path)


def pad_truncate_fixed(wav: torch.Tensor, max_samples: int) -> torch.Tensor:
    """CLAM-style fixed-length pad/truncate (zero-pad tail, truncate head)."""
    wav = wav.clone()
    cur = wav.shape[-1]
    if cur > max_samples:
        return wav[..., :max_samples]
    if cur < max_samples:
        pad = torch.zeros(*wav.shape[:-1], max_samples - cur, dtype=wav.dtype)
        return torch.cat([wav, pad], dim=-1)
    return wav


def load_mono_16k(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = T.Resample(orig_freq=sr, new_freq=SR)(wav)
    return wav


def augment_shared(wav: torch.Tensor, stem: str, aug_variant: int) -> torch.Tensor:
    rng = random.Random(aug_seed(stem, aug_variant))
    return apply_audio_augment(wav, rng, SR)


@dataclass
class OutputPaths:
    singgraph_vocals: Path
    singgraph_instrumental: Path
    proposed_w2v: Path
    proposed_mert: Path
    clam_w2v: Path
    clam_mert: Path

    @classmethod
    def for_stem(cls, output_dir: Path, stem: str) -> "OutputPaths":
        return cls(
            singgraph_vocals=output_dir / "singgraph_features" / "vocals" / f"{stem}.pt",
            singgraph_instrumental=output_dir / "singgraph_features" / "instrumental" / f"{stem}.pt",
            proposed_w2v=output_dir / "proposed_features" / "wav2vec" / f"{stem}.pt",
            proposed_mert=output_dir / "proposed_features" / "mert" / f"{stem}.pt",
            clam_w2v=output_dir / "clam_features" / "wav2vec" / f"{stem}.pt",
            clam_mert=output_dir / "clam_features" / "mert" / f"{stem}.pt",
        )

    def all_exist(self, skip_singgraph: bool = False) -> bool:
        required = (
            self.proposed_w2v,
            self.proposed_mert,
            self.clam_w2v,
            self.clam_mert,
        )
        if not skip_singgraph:
            required = (
                self.singgraph_vocals,
                self.singgraph_instrumental,
                *required,
            )
        return all(p.exists() for p in required)

    def as_index_entry(self, stem: str) -> dict[str, str]:
        return {
            "stem": stem,
            "singgraph_vocals": str(self.singgraph_vocals),
            "singgraph_instrumental": str(self.singgraph_instrumental),
            "proposed_w2v": str(self.proposed_w2v),
            "proposed_mert": str(self.proposed_mert),
            "clam_w2v": str(self.clam_w2v),
            "clam_mert": str(self.clam_mert),
        }


@dataclass
class FrozenExtractorBundle:
    device: torch.device
    load_demucs: bool = True
    resampler_mert: T.Resample = field(init=False)

    w2v_xlsr_processor: Wav2Vec2FeatureExtractor = field(init=False)
    w2v_xlsr_model: Wav2Vec2Model = field(init=False)
    mert_330m_processor: Wav2Vec2FeatureExtractor = field(init=False)
    mert_330m_model: AutoModel = field(init=False)

    w2v_clam_processor: Wav2Vec2FeatureExtractor = field(init=False)
    w2v_clam_model: AutoModel = field(init=False)
    mert_95m_processor: Wav2Vec2FeatureExtractor = field(init=False)
    mert_95m_model: AutoModel = field(init=False)

    demucs_model: torch.nn.Module = field(init=False)
    demucs_sr: int = field(init=False)
    vocals_idx: int = field(init=False)

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

        logger.info("Loading Demucs (%s)", DEMUCS_MODEL)
        if self.load_demucs:
            self.demucs_model = get_model(DEMUCS_MODEL).to(dev)
            self.demucs_sr = self.demucs_model.samplerate
            self.vocals_idx = self.demucs_model.sources.index("vocals")
            self.demucs_model.eval()
            for param in self.demucs_model.parameters():
                param.requires_grad = False
        else:
            self.demucs_model = None
            self.demucs_sr = 0
            self.vocals_idx = -1

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
        """batch_wav: [B, T] on CPU float32 -> [B, seq, 1024] on CPU float32."""
        inputs = self.w2v_xlsr_processor(
            batch_wav.numpy(),
            sampling_rate=SR,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                out = self.w2v_xlsr_model(inputs.input_values.to(self.device)).last_hidden_state
        return out.float().cpu()

    def _mert_330m_last(self, batch_wav_16k: torch.Tensor) -> torch.Tensor:
        """batch_wav_16k: [B, T] CPU -> [B, seq, 1024] CPU."""
        batch_24k = self.resampler_mert(batch_wav_16k)
        inputs = self.mert_330m_processor(
            batch_24k.numpy(),
            sampling_rate=MERT_SR,
            return_tensors="pt",
            padding=False,
        )
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                out = self.mert_330m_model(inputs.input_values.to(self.device)).last_hidden_state
        return out.float().cpu()

    def _clam_all_layers(self, model, processor, wav_16k: torch.Tensor, mert: bool) -> torch.Tensor:
        """Single waveform [1, T] -> [T, 13, 768]."""
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
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                out = model(
                    **{k: v.to(self.device) for k, v in inputs.items()},
                    output_hidden_states=True,
                )
        layers = torch.stack(out.hidden_states, dim=1).squeeze(0)  # [13, T, 768]
        layers = layers.permute(1, 0, 2).contiguous()  # [T, 13, 768]
        if layers.shape[1] != 13 or layers.shape[2] != 768:
            raise ValueError(f"Unexpected CLAM layer shape: {tuple(layers.shape)}")
        return layers.float().cpu()

    def separate_demucs(self, wav_16k_mono: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return vocals and instrumental at 16 kHz mono."""
        if self.demucs_model is None:
            raise RuntimeError("Demucs model not loaded")
        mix = convert_audio(wav_16k_mono, SR, self.demucs_sr, 2)
        with torch.no_grad():
            _, sources = apply_model(
                self.demucs_model,
                mix[None].to(self.device),
                device=self.device,
                split=True,
                overlap=0.25,
                progress=False,
            )
        sources = sources[0]  # [n_stems, channels, samples]
        vocals = sources[self.vocals_idx].mean(dim=0, keepdim=True)
        instrumental = torch.zeros_like(vocals)
        for i, name in enumerate(self.demucs_model.sources):
            if name != "vocals":
                instrumental = instrumental + sources[i].mean(dim=0, keepdim=True)

        vocals_16k = convert_audio(vocals, self.demucs_sr, SR, 1)
        inst_16k = convert_audio(instrumental, self.demucs_sr, SR, 1)
        return vocals_16k.cpu(), inst_16k.cpu()

    def _windowize_4s(self, wav: torch.Tensor) -> torch.Tensor:
        """[1, T] -> [N, 4*SR] dropping trailing partial window."""
        window = SINGGRAPH_WINDOW_SEC * SR
        total = (wav.shape[-1] // window) * window
        if total == 0:
            return wav.new_zeros((0, window))
        trimmed = wav[..., :total].squeeze(0)
        return trimmed.unfold(0, window, window)

    def extract_singgraph(
        self,
        raw_wav: torch.Tensor,
        paths: OutputPaths,
        window_batch: int,
    ) -> None:
        vocals, instrumental = self.separate_demucs(raw_wav)
        vocal_windows = self._windowize_4s(vocals)
        inst_windows = self._windowize_4s(instrumental)

        vocal_feats: list[torch.Tensor] = []
        for start in range(0, len(vocal_windows), window_batch):
            batch = vocal_windows[start : start + window_batch]
            if len(batch) == 0:
                break
            vocal_feats.append(self._w2v_xlsr_last(batch))

        inst_feats: list[torch.Tensor] = []
        for start in range(0, len(inst_windows), window_batch):
            batch = inst_windows[start : start + window_batch]
            if len(batch) == 0:
                break
            inst_feats.append(self._mert_330m_last(batch))

        vocals_out = torch.cat(vocal_feats, dim=0) if vocal_feats else torch.zeros(0, 0, 1024)
        inst_out = torch.cat(inst_feats, dim=0) if inst_feats else torch.zeros(0, 0, 1024)

        save_fp16(vocals_out, paths.singgraph_vocals)
        save_fp16(inst_out, paths.singgraph_instrumental)

        del vocals, instrumental, vocal_windows, inst_windows, vocal_feats, inst_feats
        del vocals_out, inst_out

    def extract_proposed(self, aug_wav: torch.Tensor, paths: OutputPaths) -> None:
        total_samples = PROPOSED_SEC * SR
        chunk_samples = PROPOSED_CHUNK_SEC * SR
        num_chunks = total_samples // chunk_samples

        padded = pad_loop_torch(aug_wav, total_samples)
        chunks = padded.squeeze(0).unfold(0, chunk_samples, chunk_samples)  # [4, chunk_samples]
        flat_input = chunks  # [4, T]

        out_w2v = self._w2v_xlsr_last(flat_input)
        out_mert = self._mert_330m_last(flat_input)

        out_mert = F.interpolate(
            out_mert.transpose(1, 2),
            size=out_w2v.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        final_w2v = out_w2v.reshape(num_chunks * out_w2v.shape[1], out_w2v.shape[2])
        final_mert = out_mert.reshape(num_chunks * out_mert.shape[1], out_mert.shape[2])

        if final_w2v.shape != (6000, 1024):
            logger.warning("Proposed W2V shape %s (expected [6000, 1024])", tuple(final_w2v.shape))
        if final_mert.shape != (6000, 1024):
            logger.warning("Proposed MERT shape %s (expected [6000, 1024])", tuple(final_mert.shape))

        save_fp16(final_w2v, paths.proposed_w2v)
        save_fp16(final_mert, paths.proposed_mert)

        del padded, chunks, flat_input, out_w2v, out_mert, final_w2v, final_mert

    def extract_clam(self, aug_wav: torch.Tensor, paths: OutputPaths) -> None:
        fixed = pad_truncate_fixed(aug_wav, CLAM_SEC * SR)
        w2v_layers = self._clam_all_layers(
            self.w2v_clam_model, self.w2v_clam_processor, fixed, mert=False
        )
        mert_layers = self._clam_all_layers(
            self.mert_95m_model, self.mert_95m_processor, fixed, mert=True
        )

        save_fp16(w2v_layers, paths.clam_w2v)
        save_fp16(mert_layers, paths.clam_mert)

        del fixed, w2v_layers, mert_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract benchmark features (fp16) for SONICS subset.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aug-variant", type=int, default=0)
    parser.add_argument("--singgraph-window-batch", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-singgraph", action="store_true", help="Skip Demucs/SingGraph branch (debug)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_files(input_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(input_dir.glob("*.flac"))
    if limit is not None:
        files = files[:limit]
    return files


def write_report(
    output_dir: Path,
    *,
    total: int,
    ok: int,
    skipped: int,
    errors: int,
    elapsed_s: float,
    sample_shapes: dict[str, tuple] | None,
    peak_vram_gb: float | None,
) -> str:
    lines = [
        "=" * 72,
        "SONICS BENCHMARK FEATURE EXTRACTION REPORT",
        "=" * 72,
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Output dir:      {output_dir}",
        f"Total files:     {total}",
        f"Processed OK:    {ok}",
        f"Skipped:         {skipped}",
        f"Errors:          {errors}",
        f"Elapsed:         {elapsed_s:.2f}s",
    ]
    if peak_vram_gb is not None:
        lines.append(f"Peak VRAM (GB):  {peak_vram_gb:.2f}")
    if sample_shapes:
        lines.append("")
        lines.append("Sample tensor shapes (first successful file):")
        for key, shape in sample_shapes.items():
            lines.append(f"  {key}: {shape}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.input_dir, args.limit)
    if not files:
        logger.error("No .flac files found in %s", args.input_dir)
        return 1

    logger.info("Found %d files in %s", len(files), args.input_dir)

    if args.dry_run:
        for flac in files[:5]:
            paths = OutputPaths.for_stem(args.output_dir, flac.stem)
            logger.info("Would process %s -> %s", flac.name, paths.proposed_w2v.parent)
        logger.info("Dry run complete (%d files total).", len(files))
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bundle = FrozenExtractorBundle(device=device, load_demucs=not args.skip_singgraph)

    index_path = args.output_dir / "index.json"
    index: list[dict] = []
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    indexed_stems = {entry["stem"] for entry in index}

    errors_log = args.output_dir / "errors.log"
    ok = skipped = errors = 0
    sample_shapes: dict[str, tuple] | None = None
    t0 = time.perf_counter()

    for flac in tqdm(files, desc="Extracting features"):
        stem = flac.stem
        paths = OutputPaths.for_stem(args.output_dir, stem)

        if args.skip_existing and paths.all_exist(args.skip_singgraph):
            skipped += 1
            continue

        try:
            raw = load_mono_16k(flac)
            if not args.skip_singgraph:
                bundle.extract_singgraph(raw, paths, args.singgraph_window_batch)
            aug = augment_shared(raw, stem, args.aug_variant)
            bundle.extract_proposed(aug, paths)
            bundle.extract_clam(aug, paths)

            if sample_shapes is None:
                sample_shapes = {
                    "singgraph_vocals": tuple(torch.load(paths.singgraph_vocals, weights_only=True).shape),
                    "singgraph_instrumental": tuple(torch.load(paths.singgraph_instrumental, weights_only=True).shape),
                    "proposed_w2v": tuple(torch.load(paths.proposed_w2v, weights_only=True).shape),
                    "proposed_mert": tuple(torch.load(paths.proposed_mert, weights_only=True).shape),
                    "clam_w2v": tuple(torch.load(paths.clam_w2v, weights_only=True).shape),
                    "clam_mert": tuple(torch.load(paths.clam_mert, weights_only=True).shape),
                }

            if stem not in indexed_stems:
                index.append(paths.as_index_entry(stem))
                indexed_stems.add(stem)

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
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    report = write_report(
        args.output_dir,
        total=len(files),
        ok=ok,
        skipped=skipped,
        errors=errors,
        elapsed_s=elapsed,
        sample_shapes=sample_shapes,
        peak_vram_gb=peak_vram,
    )
    print(report)
    (args.output_dir / "extraction_report.txt").write_text(report, encoding="utf-8")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

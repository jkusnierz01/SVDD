from abc import ABC, abstractmethod
from pathlib import Path
import json
import random
import logging

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

from src.utils.preprocessing import pad_loop_torch
from src.preprocessing.audio_augmentation import apply_audio_augment, aug_seed


class ChunkedAudioDataset(Dataset):
    """Loads audio files, pads to total_len_sec, splits into chunks.

    Returns (chunks [num_chunks, chunk_samples], stem).
    """

    def __init__(
        self,
        files: list[tuple[Path, str]],  # (path, stem) pairs
        sample_rate: int,
        total_len_sec: int = 120,
        chunk_len_sec: int = 30,
        lowpass_cutoff_hz: int = None,
        aug_variant: int | None = None,
    ):
        self.files = files
        self.sample_rate = sample_rate
        self.total_samples = int(total_len_sec * sample_rate)
        self.chunk_samples = int(chunk_len_sec * sample_rate)
        self.num_chunks = self.total_samples // self.chunk_samples
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.aug_variant = aug_variant

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, stem = self.files[idx]
        try:
            wav, sr = torchaudio.load(path)

            if wav.shape[0] > 1:
                channel_idx = torch.randint(0, wav.shape[0], (1,)).item()
                wav = wav[channel_idx: channel_idx + 1]

            if sr != self.sample_rate:
                wav = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)

            if self.lowpass_cutoff_hz is not None:
                target_sr = self.lowpass_cutoff_hz * 2
                wav = T.Resample(self.sample_rate, target_sr)(wav)
                wav = T.Resample(target_sr, self.sample_rate)(wav)

            wav = pad_loop_torch(wav, self.total_samples)

            if self.aug_variant is not None:
                rng = random.Random(aug_seed(stem, self.aug_variant))
                wav = apply_audio_augment(wav, rng, self.sample_rate)

            chunks = wav.squeeze(0).unfold(0, self.chunk_samples, self.chunk_samples)
            return chunks, stem

        except Exception as e:
            logging.info(f"Error loading {path}: {e}")
            return torch.zeros(self.num_chunks, self.chunk_samples), "ERROR"


class BaseProcessor(ABC):
    @abstractmethod
    def preprocess(self) -> dict:
        pass


class BaseFeatureProcessor(BaseProcessor):
    """GPU engine for wav2vec2 + MERT feature extraction.

    Subclasses only need to implement collect_files() which returns
    a list of (audio_path, stem) tuples.
    """

    def __init__(
        self,
        output_dir: str,
        device: str,
        sample_rate: int,
        total_len_sec: int,
        chunk_len_sec: int,
        num_workers: int,
        batch_size: int,
        wav2vec_model_name: str = "facebook/wav2vec2-xls-r-300m",
        mert_model_name: str = "m-a-p/MERT-v1-330M",
        lowpass_cutoff_hz: int = None,
        limit_files: int = None,
        train_only: bool = False,
        apply_audio_augment: bool = False,
        aug_variant: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.device = device
        self.sample_rate = sample_rate
        self.total_len_sec = total_len_sec
        self.chunk_len_sec = chunk_len_sec
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.wav2vec_model_name = wav2vec_model_name
        self.mert_model_name = mert_model_name
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.limit_files = limit_files
        self.train_only = train_only
        self.apply_audio_augment = apply_audio_augment
        self.aug_variant = aug_variant

    @abstractmethod
    def collect_files(self) -> list[tuple[Path, str]]:
        """Return list of (audio_path, stem) pairs to process."""
        pass

    def _prepare_files(self, files: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
        from src.preprocessing.audio_augmentation import augment_stem

        if self.train_only:
            files = [(p, s) for p, s in files if s.split("_", 1)[0] == "train"]
            logging.info(f"train_only: kept {len(files)} files")

        if self.apply_audio_augment:
            files = [(p, augment_stem(s, self.aug_variant)) for p, s in files]
            logging.info(
                f"apply_audio_augment: variant={self.aug_variant}, "
                f"{len(files)} output stems"
            )

        return files

    def preprocess(self):
        w2v_out_dir = self.output_dir / "wav2vec"
        mert_out_dir = self.output_dir / "mert"
        w2v_out_dir.mkdir(parents=True, exist_ok=True)
        mert_out_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Loading wav2vec model: {self.wav2vec_model_name}")
        wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec_model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(self.wav2vec_model_name).to(self.device)

        logging.info(f"Loading MERT model: {self.mert_model_name}")
        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.mert_model_name, trust_remote_code=True
        )
        mert_model = AutoModel.from_pretrained(
            self.mert_model_name, trust_remote_code=True
        ).to(self.device)

        wav2vec_model.eval()
        mert_model.eval()

        files = self.collect_files()
        files = self._prepare_files(files)
        random.seed(12)
        random.shuffle(files)
        if self.limit_files is not None:
            files = files[: self.limit_files]
            logging.info(f"[DEBUG] limit_files={self.limit_files}, processing {len(files)} files")

        total = len(files)
        files = [
            (path, stem) for path, stem in files
            if not (
                (w2v_out_dir / f"{stem}.npy").exists()
                and (mert_out_dir / f"{stem}.npy").exists()
            )
        ]
        logging.info(f"Resuming: {total - len(files)} already done, {len(files)} remaining")

        index = []
        json_path = self.output_dir / "index.json"
        if json_path.exists() and json_path.stat().st_size > 0:
            with open(json_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            logging.info(f"Loaded existing index: {len(index)} entries")

        dataset = ChunkedAudioDataset(
            files=files,
            sample_rate=self.sample_rate,
            total_len_sec=self.total_len_sec,
            chunk_len_sec=self.chunk_len_sec,
            lowpass_cutoff_hz=self.lowpass_cutoff_hz,
            aug_variant=self.aug_variant if self.apply_audio_augment else None,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        resampler_mert = T.Resample(orig_freq=16000, new_freq=24000)

        for batch_chunks, stems in tqdm(dataloader, desc="Processing batches"):
            valid_mask = [s != "ERROR" for s in stems]
            if not any(valid_mask):
                continue

            batch_chunks = batch_chunks[torch.tensor(valid_mask)]
            clean_stems = [stems[i] for i, v in enumerate(valid_mask) if v]

            current_batch_size = len(clean_stems)
            num_chunks = batch_chunks.shape[1]
            chunk_len = batch_chunks.shape[2]

            flat_input = batch_chunks.view(-1, chunk_len).numpy()

            try:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        inputs_w2v = wav2vec_processor(
                            flat_input,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=False,
                        )
                        out_w2v = wav2vec_model(
                            inputs_w2v.input_values.to(self.device)
                        ).last_hidden_state

                        flat_input_24k = resampler_mert(torch.from_numpy(flat_input)).numpy()
                        inputs_mert = mert_processor(
                            flat_input_24k,
                            sampling_rate=24000,
                            return_tensors="pt",
                            padding=False,
                        )
                        out_mert = mert_model(
                            inputs_mert.input_values.to(self.device)
                        ).last_hidden_state

                        # Align MERT time axis to wav2vec
                        out_mert = F.interpolate(
                            out_mert.transpose(1, 2),
                            size=out_w2v.shape[1],
                            mode="linear",
                            align_corners=False,
                        ).transpose(1, 2)

                    seq_len = out_w2v.shape[1]
                    dim_w2v = out_w2v.shape[2]
                    dim_mert = out_mert.shape[2]

                    out_w2v = out_w2v.view(current_batch_size, num_chunks, seq_len, dim_w2v).float().cpu()
                    out_mert = out_mert.view(current_batch_size, num_chunks, seq_len, dim_mert).float().cpu()

                    final_w2v = out_w2v.flatten(1, 2)   # [B, 6000, 1024]
                    final_mert = out_mert.flatten(1, 2)  # [B, 6000, 1024]

                    for i, stem in enumerate(clean_stems):
                        w2v_path = w2v_out_dir / f"{stem}.npy"
                        mert_path = mert_out_dir / f"{stem}.npy"
                        np.save(w2v_path, final_w2v[i].numpy())
                        np.save(mert_path, final_mert[i].numpy())
                        index.append({"stem": stem, "wav2vec": str(w2v_path), "mert": str(mert_path)})

            except Exception as e:
                logging.info(f"Error processing batch {clean_stems}: {e}")
                continue

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved index: {json_path}")
        except Exception as e:
            logging.info(f"Failed to write index: {e}")

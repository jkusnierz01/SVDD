from src.preprocessing.base import BaseProcessor
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoModel
from torch.utils.data import Dataset, DataLoader
import json
from src.utils.preprocessing import pad_loop_torch


class ChunkedAudioDataset(Dataset):
    def __init__(self, file_paths, sample_rate, total_len_sec=120, chunk_len_sec=30):
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.total_samples = int(total_len_sec * sample_rate)
        self.chunk_samples = int(chunk_len_sec * sample_rate)

        self.num_chunks = self.total_samples // self.chunk_samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                # OR MEAN
                # wav = torch.mean(wav, dim=0, keepdim=True)
                channel_idx = torch.randint(0, wav.shape[0], (1,)).item()
                wav = wav[channel_idx : channel_idx + 1]

            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                wav = resampler(wav)


            wav = pad_loop_torch(wav, self.total_samples)

            # [1, 120s] - [4, 30s].
            # unfold tnie tensor na okna.
            chunks = wav.squeeze(0).unfold(0, self.chunk_samples, self.chunk_samples)

            # chunks shape: [4, chunk_samples]

            return chunks, path.stem

        except Exception as e:
            return torch.zeros(self.num_chunks, self.chunk_samples), "ERROR"


# now sonics but can be done to process the same way all files
# then change name to FeaturePreprocessor
class SonicsPreprocessor(BaseProcessor):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        device: str,
        sample_rate: int,
        total_len_sec: int,
        chunk_len_sec: int,
        num_workers: int,
        batch_size: int,
        wav2vec_model_name: str = "facebook/wav2vec2-xls-r-300m",
        mert_model_name: str = "m-a-p/MERT-v1-330M",
    ):
        self.device = device
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.total_len_sec = total_len_sec
        self.chunk_len_sec = chunk_len_sec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wav2vec_model_name = wav2vec_model_name
        self.mert_model_name = mert_model_name

    def preprocess(self):
        w2v_out_dir = self.output_dir / "wav2vec"
        mert_out_dir = self.output_dir / "mert"
        w2v_out_dir.mkdir(parents=True, exist_ok=True)
        mert_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"loading procesor & model for speech... {self.wav2vec_model_name}")
        wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.wav2vec_model_name
        )
        wav2vec_model = Wav2Vec2Model.from_pretrained(self.wav2vec_model_name).to(
            self.device
        )

        print(f"loading procesor & model for music... {self.mert_model_name}")
        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.mert_model_name, trust_remote_code=True, 
        )
        mert_model = AutoModel.from_pretrained(
            self.mert_model_name, trust_remote_code=True
        ).to(self.device)

        wav2vec_model.eval()
        mert_model.eval()

        files = list(self.input_dir.rglob("*.flac"))
        dataset = ChunkedAudioDataset(
            file_paths=files,
            sample_rate=self.sample_rate,
            total_len_sec=self.total_len_sec,
            chunk_len_sec=self.chunk_len_sec,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        index = []
        for batch_chunks, filenames in tqdm(dataloader, desc="Processing Batches"):
            """
            potencjalnie do dodania attention mask wskazujące padding zerami do wav2vec i mert.
            obecnie padduje zerami ale dla modeli cały kawałek jest traktowany jako "normalny"
            """
            valid_mask = [f != "ERROR" for f in filenames]
            if not any(valid_mask):
                continue

            # [batch_size, 4, chunks]
            batch_chunks = batch_chunks[valid_mask]
            clean_filenames = [filenames[i] for i, v in enumerate(valid_mask) if v]

            current_batch_size = len(clean_filenames)
            num_chunks = batch_chunks.shape[1]
            chunk_len = batch_chunks.shape[2]

            # [Batch, 4, Len] -> [Batch * 4, Len]
            flat_input = batch_chunks.view(-1, chunk_len).numpy()

            try:
                with torch.no_grad():
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        
                        inputs_w2v = wav2vec_processor(
                            flat_input,
                            sampling_rate=self.sample_rate,
                            return_tensors="pt",
                            padding=False,
                        )
                        inputs_w2v = inputs_w2v.input_values.to(self.device)


                        out_w2v = wav2vec_model(inputs_w2v).last_hidden_state
                        # out_w2v shape: [Batch * 4, Seq_Len_Per_Chunk, 768]

                        inputs_mert = mert_processor(
                            flat_input,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=False,
                        )
                        inputs_mert = inputs_mert.input_values.to(self.device)

                        out_mert = mert_model(inputs_mert).last_hidden_state
                        # out_mert shape: [Batch * 4, Seq_Len_Per_Chunk, 768]

                    seq_len = out_w2v.shape[1]
                    hidden_dim = out_w2v.shape[2]


                    out_w2v = (
                        out_w2v.view(current_batch_size, num_chunks, seq_len, hidden_dim)
                        .float()
                        .cpu()
                    )
                    out_mert = (
                        out_mert.view(current_batch_size, num_chunks, seq_len, hidden_dim)
                        .float()
                        .cpu()
                    )


                    final_w2v = out_w2v.flatten(1, 2)
                    final_mert = out_mert.flatten(1, 2)

                    
                    for i, stem in enumerate(clean_filenames):
                        w2v_path = w2v_out_dir / f"{stem}.pt"
                        mert_path = mert_out_dir / f"{stem}.pt"
                        torch.save(final_w2v[i], w2v_path)
                        torch.save(final_mert[i], mert_path)

                        index.append(
                            {"stem": stem, "wav2vec": str(w2v_path), "mert": str(mert_path)}
                        )

            except Exception as e:
                print(f"Error processing batch {clean_filenames}: {e}")
                continue
        try:
            json_path = self.output_dir / "index.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(index, jf, indent=2, ensure_ascii=False)
            print(f"Saved index files: {json_path}")
        except Exception as e:
            print(f"Failed to write index files: {e}")

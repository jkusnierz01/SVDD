from huggingface_hub import snapshot_download
import logging
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yt_dlp
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import subprocess
import shutil as _shutil


logging.basicConfig(level=logging.INFO)
logging.getLogger("yt_dlp").setLevel(logging.ERROR)


class _YtDlpLogger:
    def debug(self, msg):
        return

    def warning(self, msg):
        return

    def error(self, msg):
        logging.error(msg)


def download_from_hf(dir_path: Path):
    logging.info("Downloading from huggingface")
    snapshot_download(
        repo_id="awsaf49/sonics",
        repo_type="dataset",
        local_dir=str(dir_path),
    )


def extract_fake_archives(hf_dir: Path):
    zips = list(hf_dir.rglob("*.zip"))
    if not zips:
        return

    logging.info(f"Extracting {len(zips)} zip archives")
    for zip_path in tqdm(zips, desc="Extracting zips"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)
        zip_path.unlink()


def convert_to_flac(src_path: Path, dest_path: Path):
    if dest_path.exists():
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.suffix.lower() == ".flac":
        _shutil.copy2(src_path, dest_path)
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-acodec",
        "flac",
        str(dest_path),
    ]
    subprocess.run(cmd, check=True)
    
def process_single_fake_row(row, file_map, audio_dir):
    csv_filename = str(row["filename"]).strip()
    src_path = file_map.get(csv_filename)
    
    if not src_path:
        return None, {"kind": "fake", "original_filename": csv_filename, "status": "missing"}

    split = str(row["split"]).strip().lower()
    # Bezpieczne pobieranie pól (niektóre mogą być puste/float NaN)
    lyrics = str(row.get('lyrics', ''))
    algorithm = str(row.get('algorithm', ''))
    mood = str(row.get('mood', ''))
    genre = str(row.get('genre', ''))
    topic = str(row.get('topic', ''))
    source = str(row.get('source', ''))
    duration = str(row.get('duration', ''))
    bit_rate = str(row.get('bit_rate', ''))
    style = str(row.get('style', ''))
    label = str(row.get('label', ''))
    spoof_type = "deepfake"

    new_filename = f"{split}_{csv_filename}_{source}_{genre}_{spoof_type}.flac"
    dest_path = audio_dir / new_filename

    try:
        if not dest_path.exists():
            convert_to_flac(src_path, dest_path)
            log_entry = {"kind": "fake", "filename": new_filename, "status": "converted"}
        else:
            log_entry = {"kind": "fake", "filename": new_filename, "status": "skipped"}

        meta_entry = {
            "dataset": "Sonics",
            "filename": new_filename,
            "filepath": str(dest_path),
            "set": split,
            "duration": duration,
            "bit_rate": bit_rate,
            "topic": topic,
            "style": style,
            "label": label,
            "lyrics": lyrics,
            "algorithm": algorithm,
            "mood": mood,
            "spoof": spoof_type,
            "original_filename": csv_filename,
        }
        return meta_entry, log_entry

    except Exception as e:
        return None, {"kind": "fake", "filename": new_filename, "status": "error", "error": str(e)}
    

def process_fake_songs(hf_dir: Path, audio_dir: Path, metadata_list: list, log_list: list, workers):
    csv_path = hf_dir / "fake_songs.csv"
    search_root = hf_dir / "fake_songs"
    
    df = pd.read_csv(csv_path).fillna("")
    extract_fake_archives(hf_dir)

    if not search_root.exists():
        return

    file_map = {f.stem: f for f in search_root.rglob("*.mp3")}

    records = df.to_dict(orient="records")
    
    logging.info(f"Processing {len(records)} | num workers {workers}")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_fake_row, row, file_map, audio_dir) for row in records]
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing fake songs"):
            meta, log = fut.result()
            if meta:
                metadata_list.append(meta)
            if log:
                log_list.append(log)

    if search_root.exists():
        _shutil.rmtree(search_root)

    
def process_real_single_row(row: dict, audio_dir: Path):
    yt_id = row.get("youtube_id")
    if not yt_id or pd.isna(yt_id):
        return None, None

    url = f"https://www.youtube.com/watch?v={yt_id}"
    
    split = str(row.get("split", "")).strip().lower()
    singer = str(row.get("artist", ""))
    title = str(row.get("title", ""))
    year = str(row.get("year", ""))
    duration = str(row.get("duration", ""))
    artist_overlap = str(row.get("artist_overlap", ""))
    spoof_type = "bonafide"

    safe_singer = singer.replace("/", "_").replace(" ", "_")
    safe_title = title.replace("/", "_").replace(" ", "_")[:20]
    filename = f"{split}_{safe_singer}_{safe_title}_{spoof_type}.flac"
    filepath = audio_dir / filename
    filepath_str = str(filepath)
    
    filepath_no_ext = str(filepath).rsplit('.', 1)[0]

    meta = {
        "dataset": "Sonics",
        "filename": filename,
        "filepath": filepath_str,
        "set": split,
        "youtube_id": yt_id,
        "year": year,
        "duration": duration,
        "artist_overlap": artist_overlap,
        "spoof": spoof_type,
        "title": title,
        "singer": singer,
        "url": url,
    }

    if filepath.exists():
        return meta, {"kind": "real", "url": url, "filename": filename, "status": "skipped"}

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "flac",
        }],
        "outtmpl": filepath_no_ext,
        
        "cookiefile": "cookies.txt",
        "sleep_interval": 10,
        "max_sleep_interval": 20,
        "ignoreerrors": True,
        
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _YtDlpLogger(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return meta, {"kind": "real", "url": url, "filename": filename, "status": "success"}
    except Exception as e:
        return None, {"kind": "real", "url": url, "status": "error", "error_msg": str(e)}


def process_sonics(hf_dir: Path, output_dir: Path, workers: int = 4):
    audio_dir = output_dir / "downloads"
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_list: list[dict] = []
    log_list: list[dict] = []

    process_fake_songs(hf_dir=hf_dir, audio_dir=audio_dir, metadata_list=metadata_list, log_list=log_list, workers=workers)

    real_csv_path = hf_dir / "real_songs.csv"
    if real_csv_path.exists():
        df_real = pd.read_csv(real_csv_path).fillna("")
        rows = df_real.to_dict(orient="records")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_real_single_row, row, audio_dir) for row in rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading real songs"):
                meta, log = fut.result()
                if meta:
                    metadata_list.append(meta)
                if log:
                    log_list.append(log)
    else:
        logging.info(f"No real_songs.csv found at: {real_csv_path}")

    metadata_path = audio_dir / "metadata.json"
    log_path = audio_dir / "log.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_list, f, indent=4, ensure_ascii=False)

    logging.info(f"Metadata saved to {metadata_path}")
    logging.info(f"Log file saved to {log_path}")
    
    
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", required=True, help="path to dir where HF snapshot + processed data will be stored")
    parser.add_argument("--download",  action="store_true", help="download dataset snapshot from HF")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel download threads")
    
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_from_hf(out_dir)

    process_sonics(hf_dir=out_dir, output_dir=out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
        
    
    
        
    
    
    
    
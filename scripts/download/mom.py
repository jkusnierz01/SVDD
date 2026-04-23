import csv
import subprocess
import fsspec
from pathlib import Path
from datasets import load_dataset, Audio

OUTPUT_DIR = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM")
BF_DIR = OUTPUT_DIR / "bonafide"
DF_DIR = OUTPUT_DIR / "deepfake"

BF_DIR.mkdir(parents=True, exist_ok=True)
DF_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = "/net/people/plgrid/plgjedrzejkusnierz/big_storage/SVDD/audio-data/MoM/real_songs.csv"
COOKIES_PATH = "/net/people/plgrid/plgjedrzejkusnierz/big_storage/SVDD/scripts/download/cookies.txt"
MAX_SAMPLES = 500

DF_KEYWORDS = ["riffusion", "yue", "voice_clone", "voice", "suno_1", "suno_3", "suno_4"]

def download_deepfakes():
    dataset_stream = load_dataset("anonymous2212/MoM-CLAM-dataset", split="validation", streaming=True)
    dataset_stream = dataset_stream.cast_column("audio", Audio(decode=False))

    df_count = 0
    print(f"Starting DF download to {DF_DIR}...")

    for sample in dataset_stream:
        if df_count >= MAX_SAMPLES:
            print("DF download finished.")
            break
            
        path = sample['audio']['path']
        file_name = Path(path).name
        
        if any(keyword in path.lower() for keyword in DF_KEYWORDS):
            with fsspec.open(path, "rb") as f_in:
                with open(DF_DIR / file_name, "wb") as f_out:
                    f_out.write(f_in.read())
            
            df_count += 1
            print(f"Downloaded DF ({df_count}/{MAX_SAMPLES}): {file_name}")

def download_bonafide():
    bf_count = 0
    print(f"\nStarting BF download to {BF_DIR}...")
    
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if bf_count >= MAX_SAMPLES:
                print("BF download finished.")
                break
                
            yt_id = row.get('youtube_id')
            filename = row.get('filename')
            
            if not yt_id or not filename:
                continue
                
            base_name = Path(filename).stem 
            output_template = str(BF_DIR / f"{base_name}.%(ext)s")
            youtube_url = f"https://www.youtube.com/watch?v={yt_id}"
            
            cmd = [
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", output_template,
                "--no-playlist",
                "--socket-timeout", "15",
                "--cookies", COOKIES_PATH,
                "--extractor-args", "youtube:player_client=mweb",
                "--sleep-interval", "2",
                "--max-sleep-interval", "5",
                youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                bf_count += 1
                print(f"Downloaded BF ({bf_count}/{MAX_SAMPLES}): {filename}")
            else:
                err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
                print(f"Skipped {yt_id}: {err}")

if __name__ == "__main__":
    download_deepfakes()
    download_bonafide()
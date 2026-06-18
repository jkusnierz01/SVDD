import os
import csv
import random
import requests
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download

# ================= KONFIGURACJA ŚCIEŻEK =================
OUTPUT_DIR = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM")
BF_DIR = OUTPUT_DIR / "bonafide"
DF_DIR = OUTPUT_DIR / "deepfake"

BF_DIR.mkdir(parents=True, exist_ok=True)
DF_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = "/net/people/plgrid/plgjedrzejkusnierz/big_storage/SVDD/audio-data/MoM/real_songs.csv"
COOKIES_PATH = "/net/people/plgrid/plgjedrzejkusnierz/big_storage/SVDD/scripts/download/cookies.txt"

REPO_ID = "anonymous2212/MoM-CLAM-dataset"
LIMIT_PER_CATEGORY = 100


def convert_to_mp3(input_path: Path, output_path: Path):
    """Convert audio file to mp3 using ffmpeg."""
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".mp3":
        shutil.copy2(input_path, output_path)
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Błąd konwersji {input_path} na mp3: {e}")
        raise

def download_deepfakes():
    print(f"Rozpoczynam zbalansowane pobieranie Deepfake do {DF_DIR}...")
    fs = HfFileSystem()

    # Zamiast skanować wszystko (co zabija serwer HF), uderzamy w konkretne foldery!
    target_folders = [
        "yue", 
        "udio/chunk_1", "udio/chunk_2", "udio/chunk_3", 
        "riffusion", 
        "diffrythm",
        "suno(v2_v4)_links"
    ]

    audio_categories = {
        "yue": {"count": 0, "files": []},
        "udio": {"count": 0, "files": []},
        "riffusion": {"count": 0, "files": []},
        "diffrythm": {"count": 0, "files": []}
    }
    
    suno_csv_files = []

    print("Pobieram listy plików partiami (żeby nie zawiesić serwerów Hugging Face)...")
    
    for folder in target_folders:
        folder_path = f"datasets/{REPO_ID}/{folder}"
        try:
            # Używamy ls() zamiast find() - to jest bardzo lekkie dla serwera!
            files_in_folder = fs.ls(folder_path, detail=False)
            print(f" Przeskanowano pomyślnie: {folder} (znaleziono {len(files_in_folder)} plików)")
            
            for file_path in files_in_folder:
                path_lower = file_path.lower()
                
                # Zbieramy pliki audio
                if file_path.endswith(('.wav', '.mp3', '.flac')):
                    for cat in audio_categories.keys():
                        if cat in path_lower and len(audio_categories[cat]["files"]) < LIMIT_PER_CATEGORY:
                            audio_categories[cat]["files"].append(file_path)
                            break
                            
                # Zbieramy pliki CSV od Suno
                elif "suno" in folder and file_path.endswith('.csv'):
                    suno_csv_files.append(file_path)
                    
        except Exception as e:
            print(f" Nie udało się przeskanować folderu {folder}: {e}")

    # 2. POBIERAMY ZWYKŁE PLIKI AUDIO
    for cat, data in audio_categories.items():
        if not data["files"]:
            continue
        print(f"\n--- Pobieram kategorię: {cat.upper()} ---")
        for idx, hf_path in enumerate(data["files"]):
            repo_file_path = hf_path.replace(f"datasets/{REPO_ID}/", "")
            try:
                original_filename = Path(repo_file_path).name
                # Unique name with category prefix + index, always .mp3
                unique_filename = f"{cat}_{idx:03d}_{Path(original_filename).stem}.mp3"
                target_path = DF_DIR / unique_filename

                # Download to temp dir first to avoid folder structure issues
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        filename=repo_file_path,
                        local_dir=tmpdir,
                    )

                    # Find the downloaded file and convert to mp3
                    src_path = Path(tmpdir) / repo_file_path
                    if src_path.exists():
                        convert_to_mp3(src_path, target_path)
                        data["count"] += 1
                        print(f"Pobrano [{cat}] ({data['count']}/{LIMIT_PER_CATEGORY}): {unique_filename}")
                    else:
                        print(f"Plik nie znaleziony po pobraniu: {repo_file_path}")

            except Exception as e:
                print(f"Błąd pobierania {repo_file_path}: {e}")

    # 3. OBSŁUGA SPECJALNA DLA SUNO (Z PLIKÓW CSV)
    if suno_csv_files:
        print("\n--- Pobieram kategorię: SUNO (z linków CSV) ---")
        suno_total_downloaded = 0
        limit_per_csv = LIMIT_PER_CATEGORY * 2 // len(suno_csv_files)

        for csv_idx, csv_file in enumerate(suno_csv_files):
            if suno_total_downloaded >= LIMIT_PER_CATEGORY * 2:
                break

            # Extract version from CSV filename (e.g., "suno(v2_v4)_links" → "v2_v4")
            csv_version = Path(csv_file).stem.replace("suno(", "").replace(")_links", "")
            print(f"Czytam plik linków: {csv_file} (wersja: {csv_version})")
            suno_csv_downloaded = 0

            try:
                with fs.open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)

                    for row in reader:
                        if suno_csv_downloaded >= limit_per_csv or suno_total_downloaded >= LIMIT_PER_CATEGORY * 2:
                            break

                        if not row:
                            continue

                        audio_url = row[0].strip()
                        if audio_url.startswith("http"):
                            url_filename = audio_url.split("/")[-1].split("?")[0]
                            # Include suno version in filename
                            filename = f"suno_{csv_version}_{suno_total_downloaded:03d}_{Path(url_filename).stem}.mp3"
                            out_path = DF_DIR / filename

                            if out_path.exists():
                                suno_csv_downloaded += 1
                                suno_total_downloaded += 1
                                continue

                            try:
                                resp = requests.get(audio_url, timeout=15)
                                if resp.status_code == 200:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(url_filename).suffix) as tmp:
                                        tmp.write(resp.content)
                                        tmp_path = Path(tmp.name)

                                    convert_to_mp3(tmp_path, out_path)
                                    tmp_path.unlink()

                                    suno_csv_downloaded += 1
                                    suno_total_downloaded += 1
                                    print(f"Pobrano [suno-{csv_version}] ({suno_total_downloaded}/{LIMIT_PER_CATEGORY}): {filename}")
                            except Exception as e:
                                print(f"Błąd połączenia Suno: {e}")
            except Exception as e:
                print(f"Błąd odczytu pliku CSV {csv_file}: {e}")

    print("\nSUKCES: Moduł Deepfake zakończył pracę!")

def download_bonafide():
    print(f"\nRozpoczynam pobieranie prawdziwych utworów (Bonafide) do {BF_DIR}...")
    
    rows = []
    try:
        with open(CSV_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Nie udało się otworzyć pliku CSV dla Bonafide: {e}")
        return
        
    random.seed(42)
    random.shuffle(rows)
    
    bf_count = 0
    TOTAL_BF_TARGET = 500
    
    for row in rows:
        if bf_count >= TOTAL_BF_TARGET:
            print("Pobieranie Bonafide zakończone (osiągnięto limit).")
            break
            
        yt_id = row.get('youtube_id')
        filename = row.get('filename')
        
        if not yt_id or not filename:
            continue
            
        base_name = Path(filename).stem 
        output_template = str(BF_DIR / f"{base_name}.%(ext)s")
        youtube_url = f"https://www.youtube.com/watch?v={yt_id}"
        
        # Jeśli plik mp3/wav już tam jest, nie pobierajmy ponownie
        if Path(str(BF_DIR / f"{base_name}.mp3")).exists():
            bf_count += 1
            print(f"Pomijam (już istnieje) BF ({bf_count}/{TOTAL_BF_TARGET}): {filename}")
            continue
        
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
            print(f"Pobrano BF ({bf_count}/{TOTAL_BF_TARGET}): {filename}")
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "nieznany błąd"
            print(f"Pominięto {yt_id}: {err}")

if __name__ == "__main__":
    download_deepfakes()
    # download_bonafide()
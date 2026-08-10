from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
import json
import yt_dlp
from urllib.parse import urlparse
import requests
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)

FILE_URLS = {
    "train": "https://zenodo.org/records/10893604/files/train.csv?download=1",
    "test_A": "https://zenodo.org/records/10893604/files/test_A.csv?download=1",
    "test_B": "https://zenodo.org/records/10893604/files/test_B.csv?download=1"
}

def download_file(url, dest_folder, filename=None):
    if filename is None:
        filename = os.path.basename(urlparse(url).path)
    dest_path = os.path.join(dest_folder, filename)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

def read_csvs(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        dfs = [pd.read_csv(x) for x in csvs]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p)
    return df

def process_single_row(row, output_directory):
    try:
        language = row['Language']
        model = row['Model']
        platform = row['Platform']
        url = row["Url"]
        singer = row["Singer"]
        title = row["Title"]
        test_A = row["is_test_A"]
        test_B = row["is_test_B"]
        spoof_type = row["Bonafide Or Deepfake"]
        
        if test_A:
            set_name = "TestA"
        elif test_B:
            set_name = "TestB"
        else:
            set_name = "Training"
        
        safe_title = str(title).replace('/', '_').replace(' ', '_')[:20]
        
        filename_base = f"{set_name}_{singer}_{safe_title}_{spoof_type}"
        expected_filename = f"{filename_base}.flac"
        expected_filepath = Path(output_directory) / expected_filename

        if expected_filepath.exists() and expected_filepath.stat().st_size > 1024:
            return None, None 

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(output_directory) / filename_base),
            "noplaylist": True,
            "quiet": True, 
            "no_warnings": True,
            "postprocessors": [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'flac',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        meta_entry = {
            "dataset": "WildSVDD",
            "filename": expected_filename,
            "filepath": str(expected_filepath),
            "set": set_name,
            "language": language,
            "model": model,
            "platform": platform,
            "spoof_type": spoof_type,
            "title": title,
            "singer": singer,
            "url": url,
        }

        log_entry = {
            "url": url,
            "filename": expected_filename,
            "status": "success",
        }
        return meta_entry, log_entry

    except Exception as e:
        log_entry = {
            "url": row.get("Url", "unknown"),
            "filename": expected_filename if 'expected_filename' in locals() else "unknown",
            "status": "error",
            "error_msg": str(e),
        }
        return None, log_entry

def process_wildsvdd(input_path: str, output_directory: str, max_workers: int = 4):
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    df = read_csvs(input_path=input_path)
    os.makedirs(output_directory, exist_ok=True)
    
    df_filtered = df[pd.isna(df['SingFake_Set'])].copy()

    log_list = []
    metadata_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_row, row, output_directory): index 
                   for index, row in df_filtered.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            meta, log = future.result()
            
            if meta:
                metadata_list.append(meta)
            if log:
                log_list.append(log)

    output = Path(output_directory)
    log_path = output / "download_log.json"
    metadata_path = output / "metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_list, f, indent=4, ensure_ascii=False)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--csv_dir")
    parser.add_argument("--audio_dir")
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    if args.audio_dir is None:
        args.audio_dir = args.csv_dir
    
    if args.download and args.csv_dir is None:
        parser.error("--csv_dir is required when using --download flag")
    
    if args.download:
        Path(args.csv_dir).mkdir(parents=True, exist_ok=True)
        for name, url in FILE_URLS.items():
            try:
                download_file(url, args.csv_dir)
            except Exception as e:
                logging.error(f"Error {name}: {e}")
                return
            
    if not Path(args.csv_dir).exists() or not list(Path(args.csv_dir).glob("*.csv")):
        logging.error(f"No csv files in: {args.csv_dir}")
        return
            
    process_wildsvdd(
        input_path=args.csv_dir, 
        output_directory=args.audio_dir,
        max_workers=args.workers
    )
          
if __name__ == "__main__":
    main()
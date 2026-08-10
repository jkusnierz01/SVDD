import pandas as pd
import os
import yt_dlp
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import logging


def download_singfake(input_csv_path: str, output_directory: str):
    if not Path(input_csv_path).exists():
        raise FileNotFoundError("csv path does not exist")

    df = pd.read_csv(input_csv_path, sep=",")

    os.makedirs(output_directory, exist_ok=True)
    log_list = []
    metadata_list = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["Url"]
        singer = row["Singer"]
        title = row["Title"]
        language = row['Language']
        set = row["Set"]
        spoof_type = row["Bonafide Or Spoof"]

        filename = f"{set}_{singer}_{title.replace('/', '_').replace(' ', '_')}_{spoof_type}.flac"
        filepath = os.path.join(output_directory, filename)

        if filepath.exists():
            continue
        
        ydl_opts = {
            "format": "bestaudio/best",
            "extractaudio": True,
            "audioformat": "flac",
            "outtmpl": filepath,
            "noplaylist": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            metadata_list.append(
                {
                    "dataset": "SingFake",
                    "filename": filename,
                    "filepath": filepath,
                    "set": set,
                    "spoof": spoof_type,
                    "title": title,
                    "language": language,
                    "singer": singer,
                    "url": url,
                }
            )

            log_list.append(
                {
                    "url": url,
                    "filename": filename,
                    "filepath": filepath,
                    "status": "success",
                }
            )
        except Exception as e:
            logging.error(f"Błąd przy pobieraniu {url}: {e}")
            log_list.append(
                {
                    "url": url,
                    "filename": filename,
                    "filepath": filepath,
                    "status": "error",
                    "error_msg": str(e),
                }
            )
    output = Path(output_directory)
    log_path = output / "download_log.json"
    metadata_path = output / "metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)
    logging.info(f"Metadata saved to {metadata_path}")
    
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_list, f, indent=4, ensure_ascii=False)
    logging.info(f"Log file saved to {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()
    input_csv_path = args.input
    output_path = args.output

    if args.download:
        download_singfake(input_csv_path, output_path)

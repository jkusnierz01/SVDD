import argparse
import json
import multiprocessing as mp
import os
import re

import pandas as pd
import yt_dlp


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def read_csv(csv_file):
    return pd.read_csv(csv_file)


def download_audio(url, output_path, title, bonafide_or_deepfake):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "flac",
            }
        ],
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        # Usuń 'cookiesfrombrowser': ('chrome',), jeśli nie masz Chrome lub cookies
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True, None
    except Exception as e:
        return False, str(e)


def process_row(row, output_dir, metadata_list):
    singfake_set = row.get("SingFake_Set", "")
    if not (
        pd.isna(singfake_set) or str(singfake_set).strip() == ""
    ):  # Pobieraj tylko jeśli SingFake_Set jest pusty/NaN
        return

    url = row["Url"]
    title = sanitize_filename(row["Title"])
    bonafide_or_deepfake = row["Bonafide Or Deepfake"].lower()
    output_filename = f"{bonafide_or_deepfake}_{title}.flac"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        success = True
        error = None
    else:
        success, error = download_audio(url, output_path, title, bonafide_or_deepfake)

    metadata = {**row, "file_path": output_path, "success": success, "error": error}
    metadata_list.append(metadata)


def worker(queue, output_dir, metadata_list):
    while True:
        item = queue.get()
        if item is None:
            break
        process_row(item, output_dir, metadata_list)


def main():
    """

    TO RUN:
    python download_script.py --csv_file train.csv --output_dir /mnt/data/kusnierz/audio-data/WildSVDD/downloads --workers 4


    """
    parser = argparse.ArgumentParser(
        description="Download audio from URLs in CSV where SingFake_Set is NaN."
    )
    parser.add_argument("--csv_file", default="train.csv", help="Path to the CSV file.")
    parser.add_argument(
        "--output_dir",
        default="/mnt/data/kusnierz/audio-data/WildSVDD/downloads",
        help="Directory to save files.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_csv(args.csv_file)
    filtered_rows = rows[
        rows["SingFake_Set"].isna() | (rows["SingFake_Set"] == "")
    ].to_dict("records")

    manager = mp.Manager()
    metadata_list = manager.list()
    task_queue = mp.Queue()

    for row in filtered_rows:
        task_queue.put(row)

    workers = []
    for _ in range(args.workers):
        p = mp.Process(target=worker, args=(task_queue, args.output_dir, metadata_list))
        workers.append(p)
        p.start()

    for _ in range(args.workers):
        task_queue.put(None)

    for w in workers:
        w.join()

    # Zapisz metadane do JSON
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(list(metadata_list), f, ensure_ascii=False, indent=4)

    print(
        f"Downloaded {len([m for m in metadata_list if m['success']])} files successfully."
    )
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

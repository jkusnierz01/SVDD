from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import subprocess
import os
import argparse
from pathlib import Path

# This is the script used for simulating T03 (codec set) from T02.


def simulate_codec(
    input_path, temp_path, output_path, format, bitrate="128k", codec=None
):
    input_path = str(input_path)
    temp_path = str(temp_path)
    output_path = str(output_path)
    # Convert the input file to the temporary path using ffmpeg
    if codec:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-i",
                input_path,
                "-acodec",
                codec,
                "-b:a",
                bitrate,
                "-y",
                temp_path,
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-i",
                input_path,
                "-b:a",
                bitrate,
                "-y",
                temp_path,
            ],
            check=True,
        )
    # Read the temporary file and convert it back to flac
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-i",
            temp_path,
            "-acodec",
            "flac",
            "-y",
            output_path,
        ],
        check=True,
    )

    # Remove the temporary file
    os.remove(temp_path)


def format_suffix(format):
    if format == "mp3":
        return "mp3"
    elif format == "adts":
        return "aac"
    elif format == "ogg":
        return "ogg"
    elif format == "opus":
        return "opus"
    else:
        raise ValueError(f"Invalid format {format}")


def format_codec(format):
    if format == "mp3":
        return "libmp3lame"
    elif format == "ogg":
        return "libvorbis"
    elif format == "adts":
        return "aac"
    elif format == "opus":
        return "libopus"
    else:
        raise ValueError(f"Invalid format {format}")


def worker(file_tuple):
    src_file_path, dest_file_path, format, bitrate = file_tuple
    temp_path = str(dest_file_path.with_suffix("." + format_suffix(format)))
    try:
        simulate_codec(
            src_file_path,
            temp_path,
            dest_file_path,
            format=format,
            bitrate=bitrate,
            codec=format_codec(format),
        )
    except Exception as e:
        print(f"Failed to simulate {src_file_path}")
        print(e)


def process_audio_files(
    src_folder: Path,
    dest_folder: Path,
    format: str = "mp3",
    bitrate: str = "128k",
    max_workers: int = None,
):
    file_list = []
    suffix = "T03"
    dest_folder.mkdir(parents=True, exist_ok=True)

    filelist = list(src_folder.glob("T02*"))
    for filepath in filelist:
        filename = filepath.stem

        parts = filename.split("_")
        parts[0] = f"{suffix}_{format_suffix(format)}"
        new_filename = "_".join(parts)
        full_new_filename = dest_folder / (new_filename + ".flac")
        if full_new_filename.exists():
            continue
        
        file_list.append((filepath, full_new_filename, format, bitrate))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(worker, file_list),
                total=len(file_list),
                desc=f"Processing {src_folder}",
            )
        )


def main():
    parser = argparse.ArgumentParser(
        "Simulate different codecs from T02 SingFake audio samples"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="directory containing T02 files to be processed",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default=None,
        help="directory containing T02 files to be processed",
    )
    parser.add_argument(
        "--num_workers",
        required=False,
        default=8,
        type=int,
        help="directory containing T02 files to be processed",
    )

    args = parser.parse_args()
    num_workers = args.num_workers
    input_dir = Path(args.input_dir)
    if not args.output_dir:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir)

    audio_formats = [
        ["mp3", "128k"],
        ["ogg", "64k"],
        ["opus", "64k"],
        ["adts", "64k"],
    ]  # write audio formats you want
    assert audio_formats is not None, "You must specify audio codec formats!"

    for audio_format_tuple in audio_formats:
        audio_format = audio_format_tuple[0]
        bitrate = audio_format_tuple[1]
        print("Processing " + audio_format + "...")

        process_audio_files(
            input_dir,
            output_dir,
            format=audio_format,
            bitrate=bitrate,
            max_workers=num_workers,
        )


if __name__ == "__main__":
    main()

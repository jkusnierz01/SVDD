import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from huggingface_hub import login
import torch

"""
---------------------------------------------------
1. SCRIPT RUNS DEMUCS AND SEPARATES VOCALS FROM MUSIC
2. THEN VAD MODEL IS USED TO CREATE A FILE WHICH INDICATES PARTS WITH VOCALS FROM NOISE/SILENCE
---------------------------------------------------
"""



_original_load = torch.load

def strict_load_bypass(*args, **kwargs):
    kwargs['weights_only'] = False 
    return _original_load(*args, **kwargs)

torch.load = strict_load_bypass

DEFAULT_OUTPUT_PATH = Path("/mnt/data/kusnierz/audio-data/SingFake/data_after_separation")
VAD_TOKEN = os.getenv("VAD_TOKEN")

assert VAD_TOKEN is not None, ("You must provide an auth token to use PyAnnote VAD pipeline.")

parser = argparse.ArgumentParser(description="Separate vocals and run VAD.")
parser.add_argument("--download_dump_dir", help="directory where the download dump files are stored")
args = parser.parse_args()
download_dump_dir = Path(args.download_dump_dir)

DEFAULT_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------
# 1) SEPARATION (Demucs)
# ---------------------------------------------------------

all_files = sorted(download_dump_dir.iterdir())
files_to_process = []

for file in all_files:
    output_folder = DEFAULT_OUTPUT_PATH / "mdx_extra" / file.stem
    vocals_path = output_folder / "vocals.wav"
    if not vocals_path.exists():
        files_to_process.append(file)

if files_to_process:
    BATCH_SIZE = 50 
    chunks = [files_to_process[i:i + BATCH_SIZE] for i in range(0, len(files_to_process), BATCH_SIZE)]

    print(f"Found {len(files_to_process)} files")

    for i, chunk in enumerate(chunks):
        
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", "mdx_extra",
            "-d", "cuda",      
            "-j", "4",         
            "-o", str(DEFAULT_OUTPUT_PATH)
        ] + chunk              

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in batch {i+1}: {e}")
else:
    print("All files ready")


# ---------------------------------------------------------
# 1) VAD
# ---------------------------------------------------------
print("\n --- 2): Voice Activity Detection (VAD) ---")

login(token=VAD_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model.from_pretrained("pyannote/segmentation").to(device)
pipeline = VoiceActivityDetection(segmentation=model)

HYPER_PARAMETERS = {
    "onset": 0.5,
    "offset": 0.5,
    "min_duration_on": 3.0,
    "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)
pipeline.to(device)

for file in tqdm(files_to_process, desc="Running VAD"):
    output_folder = DEFAULT_OUTPUT_PATH / "mdx_extra" / file.stem
    vocals_path = output_folder / "vocals.wav"
    vad_path = vocals_path.with_suffix(".vad")

    if vad_path.exists():
        continue

    if vocals_path.exists():
        try:
            vad = pipeline(str(vocals_path))
            with open(str(vad_path), "w") as f:
                f.write(str(vad))
        except Exception as e:
            print(f"Error VAD: {file.name}: {e}")
    else:
        print(f"Not found {vocals_path}, skipping VAD.")
import os, sys
from tqdm import tqdm
import demucs.separate
import subprocess
from pyannote.audio import Model, Inference
from pathlib import Path
from huggingface_hub import login

authtoken = ""
assert authtoken is not None, (
    "You must provide an auth token to use PyAnnote VAD pipeline."
)

print("before model...")
login(token="")
model = Model.from_pretrained("pyannote/segmentation")
print("model loaded :)")
from pyannote.audio.pipelines import VoiceActivityDetection

pipeline = VoiceActivityDetection(segmentation=model)

HYPER_PARAMETERS = {
    # onset/offset activation thresholds (probabilities)
    "onset": 0.5,
    "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 3.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)


def run_vad(file_path):
    vad = pipeline(file_path)
    vad = str(vad)
    # note that this step only generates a "*.vad" file
    # this file will be of the same name but different extension (.vad)
    with open(file_path.replace(".wav", ".vad"), "w") as f:
        f.write(vad)


if len(sys.argv) != 2:
    print("Usage: python dataset-separate.py <download_dump_dir>")
    print(
        "The 'download_dump_dir' is the directory where the download dump files are stored."
    )
    print("Expecting results in the format produced by dataset-download.py")
    exit(1)

download_dump_dir = Path(sys.argv[1])

default_output_dir = Path(
    "/mnt/data/kusnierz/audio-data/SingFake/data_after_separation"
)
default_output_dir.mkdir(exist_ok=True, parents=True)

for file in tqdm(download_dump_dir.iterdir(), desc="Separate Vocals"):
    output_folder = Path("mdx_extra") / file.stem
    vocals_path = output_folder / "vocals.wav"
    vad_path = vocals_path.with_suffix(".vad")

    if vocals_path.exists() and vad_path.exists():
        print(f"{vocals_path} już przetworzony, pomijam...")
        continue

    try:
        subprocess.run(
            [
                "demucs",
                "--two-stems=vocals",
                "-n",
                "mdx_extra",
                "-o",
                str(default_output_dir),
                str(file),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("Przerwano przez użytkownika.")
        break
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas separacji: {e}")
        continue

    full_path = default_output_dir / vocals_path
    if vocals_path.exists():
        run_vad(str(full_path))
    else:
        print(f"Nie znaleziono {full_path}, pomijam VAD.")

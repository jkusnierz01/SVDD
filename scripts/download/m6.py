from huggingface_hub import snapshot_download
from pathlib import Path
import logging

OUTPUT_DIR = Path("/net/people/plgrid/plgjedrzejkusnierz/scratch/data/M6")

def download_from_hf(dir_path: Path):
    logging.info("Downloading from huggingface")
    snapshot_download(
        repo_id="yl7622/M6",
        repo_type="dataset",
        local_dir=str(dir_path),
    )
    
    
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    download_from_hf(dir_path)
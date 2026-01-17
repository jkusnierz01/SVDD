import hydra
from omegaconf import DictConfig
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path
from hydra.utils import instantiate
import rootutils

ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing on: {device}")

    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    processor = instantiate(
        cfg.preprocessing,
        input_dir=str(cfg.paths.input_dir),
        output_dir=str(cfg.paths.output_dir),
        device=str(device),
    )
    processor.preprocess()

if __name__ == "__main__":
    main()
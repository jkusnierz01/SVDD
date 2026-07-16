import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
from hydra.utils import instantiate
import rootutils

ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)


@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing on: {device}")

    out_dir = cfg.preprocessing.get("output_dir", None)
    if not out_dir and cfg.get("paths") and cfg.paths.get("output_dir"):
        out_dir = cfg.paths.output_dir
        
    if not out_dir:
        raise ValueError("output_dir must be specified either in preprocessing config or paths config")
        
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    processor = instantiate(
        cfg.preprocessing,
        output_dir=str(out_dir),
        device=str(device),
    )
    processor.preprocess()


if __name__ == "__main__":
    main()

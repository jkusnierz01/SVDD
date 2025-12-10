import torch
from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb

@hydra.main(config_path="configs", config_name="eval.yaml", version_base="1.1")
def main(cfg: DictConfig):
    assert cfg.ckpt_path
    
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)



if __name__ == "__main__":
    main()
    

    
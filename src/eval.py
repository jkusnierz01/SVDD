import torch
from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb
import rootutils

ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

@hydra.main(config_path="configs", config_name="eval.yaml", version_base="1.1")
def main(cfg: DictConfig):
    assert cfg.cpk_path  
    
    datamodule = instantiate(cfg.data)
    logger = instantiate(cfg.logger) if cfg.logger else None
    trainer = instantiate(cfg.trainer, logger=logger)
    
    model = instantiate(cfg.model)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.cpk_path, weights_only=False)
    wandb.finish()


if __name__ == "__main__":
    main()
    

    
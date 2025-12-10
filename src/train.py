import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import wandb
import rootutils

ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    model = instantiate(cfg.model)
    
    logger = instantiate(cfg.logger)
    callbacks = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate(cb_conf))
    trainer = instantiate(cfg.trainer, logger = logger, callbacks=callbacks)
    
    datamodule = instantiate(cfg.data)
    trainer.fit(model=model, datamodule=datamodule)
    
    wandb.finish()


if __name__ == "__main__":
    main()

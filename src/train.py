import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import wandb
import rootutils

ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    logger = instantiate(cfg.logger)

    callbacks = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate(cb_conf))
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    # Log full config to wandb so every run has complete reproducibility info
    if logger and hasattr(logger, 'experiment'):
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()

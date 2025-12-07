import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from utils.utils import create_wandb_logger
import wandb


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg: DictConfig):
    model = instantiate(cfg.model)
    experiment_name = cfg.experiment_name
    epochs = cfg.training.epochs
    wandb_logger_base = create_wandb_logger(experiment_name)
    trainer = Trainer(
        max_epochs=epochs,
        logger=wandb_logger_base,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=5
    )
    datamodule = instantiate(cfg.datamodule)

    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()

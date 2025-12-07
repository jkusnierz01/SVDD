from lightning.pytorch.loggers import WandbLogger

def create_wandb_logger(experiment_name, project="SVDD"):
    """Create a new WandB logger for each experiment"""
    return WandbLogger(
        project=project,
        name=experiment_name,
        log_model="all",
        dir="./wandb_logs"
    )
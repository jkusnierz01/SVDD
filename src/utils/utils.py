from lightning.pytorch.loggers import WandbLogger
import wandb
import torch
import matplotlib.pyplot as plt

def create_wandb_logger(experiment_name, project="SVDD"):
    """Create a new WandB logger for each experiment"""
    return WandbLogger(
        project=project,
        name=experiment_name,
        log_model="all",
        dir="./wandb_logs"
    )
    
def wandb_log_cm(cm: torch.Tensor, key: str):
    if wandb.run is None:
        return

    cm = cm.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    wandb.log({key: wandb.Image(fig)})
    plt.close(fig)
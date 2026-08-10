import time
import lightning as L


class SlurmProgressCallback(L.Callback):
    """Plain-text progress logger for SLURM jobs (no ANSI / carriage returns)."""

    def __init__(self):
        self._epoch_start = time.time()
        self._n_batches = 0

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._epoch_start = time.time()
        self._n_batches = trainer.num_training_batches
        print(f"\n[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] training — {self._n_batches} batches", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self._n_batches:
            loss = trainer.callback_metrics.get("train/loss", float("nan"))
            print(f"  batch {batch_idx + 1}/{self._n_batches}  loss={loss:.4f}", flush=True)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        elapsed = time.time() - self._epoch_start
        m = trainer.callback_metrics
        parts = [f"{k}={v:.4f}" for k, v in sorted(m.items())]
        label = "sanity-check" if trainer.sanity_checking else f"Epoch {trainer.current_epoch + 1}"
        print(f"[{label}] {' | '.join(parts)} | time={elapsed:.0f}s", flush=True)

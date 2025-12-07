from mamba_ssm.modules.mamba_simple import Mamba
import lightning as L
import torch
import torch.nn as nn
import torchaudio
import torchmetrics


class AudioDeepfakeModel(L.LightningModule):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        lr: float,
    ):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')
        
        self.spectrogram_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels
        )
        self.linear = nn.Linear(n_mels, d_model)
        self.mamba_layer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.linear_2 = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        if x.dim() == 3:
            x = x.mean(dim=1)
        # print(x.shape)
        
        mel_spec = self.spectrogram_layer(x)
        mel_spec = torch.log(mel_spec + 1e-9) 
        mel_spec = mel_spec.transpose(1, 2)
        # print(mel_spec.shape)
        x = self.linear(mel_spec)
        x = self.mamba_layer(x)
        
        if lengths is not None:
            hop_length = 512  
            mel_lengths = lengths // hop_length
            mel_lengths = mel_lengths.clamp(min=1, max=x.size(1))  # zabezpieczenie
            
            batch_idx = torch.arange(x.size(0), device=x.device)
            last = x[batch_idx, mel_lengths - 1, :]  # (B, d_model)
        else:
            last = x[:, -1, :]
        
        logit = self.linear_2(last)
        return logit

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer 

    def training_step(self, train_batch, batch_idx):
        x, y_true, x_len = train_batch
        y_pred = self.forward(x, lengths=x_len).squeeze(-1) 
        loss = self.loss_fn(y_pred,y_true)
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true, x_len = val_batch
        y_pred = self.forward(x, lengths=x_len).squeeze(-1) 
        loss = self.loss_fn(y_pred,y_true)
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, y_true)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

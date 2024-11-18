import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav
from models.resunet import ResUNet30
import numpy as np
import torchaudio.transforms as T
import torch.nn.functional as F

class SpeakerSeparation(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()

        self.ss_model = ResUNet30(
            input_channels=1,
            output_channels=1,
            condition_size=256
        )

        # Initialize speaker encoder on CPU explicitly
        self.speaker_encoder = VoiceEncoder().cpu()

        self.learning_rate = learning_rate

        # Add metrics
        self.train_step_outputs = []
        self.validation_step_outputs = []

        # STFT for signal-to-noise ratio calculation
        self.stft = T.Spectrogram(
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            power=2
        )

    def forward(self, x):
        return self.ss_model(x)

    def _compute_metrics(self, separated, target):
        # Calculate SNR
        noise = separated - target
        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        # Calculate SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
        separated_norm = separated - torch.mean(separated, dim=-1, keepdim=True)
        target_norm = target - torch.mean(target, dim=-1, keepdim=True)

        # Normalize to zero mean and unit variance
        separated_norm = separated_norm / (torch.norm(separated_norm, dim=-1, keepdim=True) + 1e-8)
        target_norm = target_norm / (torch.norm(target_norm, dim=-1, keepdim=True) + 1e-8)

        # SI-SNR calculation
        s_target = torch.sum(separated_norm * target_norm, dim=-1, keepdim=True) * target_norm
        e_noise = separated_norm - s_target

        si_snr = 20 * torch.log10(torch.norm(s_target, dim=-1) / (torch.norm(e_noise, dim=-1) + 1e-8))

        # L1 Loss
        l1_loss = F.l1_loss(separated, target)

        return {
            'snr': snr.mean(),
            'si_snr': si_snr.mean(),
            'l1_loss': l1_loss
        }

    def _shared_step(self, batch, batch_idx, step_type='train'):
        # Move the speaker encoder to CPU (since it's having issues with MPS)
        self.speaker_encoder = self.speaker_encoder.cpu()

        # Get speaker embeddings
        if isinstance(batch['speaker_wav'][0], str):
            embeddings = []
            for wav in batch['speaker_wav']:
                # Process on CPU
                wav_data = preprocess_wav(wav)
                # Keep on CPU for embedding
                with torch.no_grad():
                    embedding = self.speaker_encoder.embed_utterance(wav_data)
                # Only move the final embedding to the target device
                embeddings.append(torch.from_numpy(embedding).float().to(self.device))
            embeddings = torch.stack(embeddings)
        else:
            embeddings = []
            for wav in batch['speaker_wav']:
                # Move to CPU for processing
                wav_cpu = wav.cpu().numpy()
                with torch.no_grad():
                    embedding = self.speaker_encoder.embed_utterance(wav_cpu)
                # Move only the embedding to target device
                embeddings.append(torch.from_numpy(embedding).float().to(self.device))
            embeddings = torch.stack(embeddings)

        # Prepare input (everything else should already be on the correct device)
        input_dict = {
            'mixture': batch['mixture'].unsqueeze(1).to(self.device),
            'condition': embeddings.to(self.device),
        }

        # Forward pass
        output_dict = self.ss_model(input_dict)
        separated = output_dict['waveform'].squeeze(1)

        # Calculate metrics
        metrics = self._compute_metrics(separated, batch['target'].to(self.device))

        # Log metrics
        for metric_name, value in metrics.items():
            self.log(
                f'{step_type}_{metric_name}',
                value,
                on_step=(step_type == 'train'),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=len(batch['mixture'])  # Add this line
            )

        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, batch_idx, 'train')
        self.train_step_outputs.append(metrics)
        return metrics['l1_loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, batch_idx, 'val')
        self.validation_step_outputs.append(metrics)
        return metrics['l1_loss']

    def on_train_epoch_end(self):
        # Calculate epoch metrics
        metrics = {
            'train_epoch_snr': torch.stack([x['snr'] for x in self.train_step_outputs]).mean(),
            'train_epoch_si_snr': torch.stack([x['si_snr'] for x in self.train_step_outputs]).mean(),
            'train_epoch_l1_loss': torch.stack([x['l1_loss'] for x in self.train_step_outputs]).mean()
        }

        # Log epoch metrics
        self.log_dict(metrics, prog_bar=True)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        # Calculate epoch metrics
        metrics = {
            'val_epoch_snr': torch.stack([x['snr'] for x in self.validation_step_outputs]).mean(),
            'val_epoch_si_snr': torch.stack([x['si_snr'] for x in self.validation_step_outputs]).mean(),
            'val_epoch_l1_loss': torch.stack([x['l1_loss'] for x in self.validation_step_outputs]).mean()
        }

        # Log epoch metrics
        self.log_dict(metrics, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_epoch_l1_loss",
                "frequency": 1
            },
        }
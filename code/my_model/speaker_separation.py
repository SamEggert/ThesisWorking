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
        self.save_hyperparameters()

        # Initialize models
        self.ss_model = ResUNet30(
            input_channels=1,
            output_channels=1,
            condition_size=256
        )

        # Initialize speaker encoder
        self.speaker_encoder = None  # Will initialize in setup()

        self.learning_rate = learning_rate
        self.train_step_outputs = []
        self.validation_step_outputs = []

        # STFT for signal-to-noise ratio calculation
        self.stft = T.Spectrogram(
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            power=2
        )

    def setup(self, stage=None):
        """Initialize speaker encoder during setup phase"""
        if self.speaker_encoder is None:
            self.speaker_encoder = VoiceEncoder().to(self.device)
            self.speaker_encoder.eval()  # Always in eval mode

    def forward(self, x):
        return self.ss_model(x)

    def _get_speaker_embedding(self, wav_input):
        """Helper function to get speaker embeddings"""
        if isinstance(wav_input, str):
            wav_data = preprocess_wav(wav_input)
        else:
            wav_data = wav_input.cpu().numpy()

        with torch.no_grad():
            embedding = self.speaker_encoder.embed_utterance(wav_data)
        return torch.from_numpy(embedding).float()


    def _shared_step(self, batch, batch_idx, step_type='train'):
        try:
            # Process embeddings
            embeddings = []
            for wav in batch['speaker_wav']:
                embedding = self._get_speaker_embedding(wav)
                embeddings.append(embedding)
            embeddings = torch.stack(embeddings).to(self.device)

            # Move input data to device
            mixture = batch['mixture'].unsqueeze(1).to(self.device)
            target = batch['target'].to(self.device)

            # Forward pass
            output_dict = self.ss_model({
                'mixture': mixture,
                'condition': embeddings,
            })
            separated = output_dict['waveform'].squeeze(1)

            # Calculate L1 loss (primary training objective)
            loss = F.l1_loss(separated, target)

            # Calculate additional metrics
            with torch.no_grad():
                snr_val = self._compute_snr(separated, target)
                si_snr_val = self._compute_si_snr(separated, target)

                metrics = {
                    'l1_loss': loss.item(),
                    'snr': snr_val.item(),
                    'si_snr': si_snr_val.item()
                }

            # Log metrics
            self.log(f'{step_type}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            for metric_name, value in metrics.items():
                if metric_name != 'l1_loss':  # Already logged the loss
                    self.log(
                        f'{step_type}_{metric_name}',
                        value,
                        on_step=(step_type == 'train'),
                        on_epoch=True,
                        prog_bar=True
                    )

            return loss, metrics

        except Exception as e:
            print(f"Error in {step_type}_step: {str(e)}")
            raise e

    def _compute_snr(self, separated, target):
        """Compute SNR for a batch of samples"""
        noise = separated - target
        signal_power = torch.mean(target ** 2, dim=-1)
        noise_power = torch.mean(noise ** 2, dim=-1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr.mean()

    def _compute_si_snr(self, separated, target):
        """Compute SI-SNR for a batch of samples"""
        # Remove mean of each sample
        separated_norm = separated - torch.mean(separated, dim=-1, keepdim=True)
        target_norm = target - torch.mean(target, dim=-1, keepdim=True)

        # Normalize to zero mean and unit variance
        separated_norm = separated_norm / (torch.norm(separated_norm, dim=-1, keepdim=True) + 1e-8)
        target_norm = target_norm / (torch.norm(target_norm, dim=-1, keepdim=True) + 1e-8)

        # Calculate s_target
        s_target = torch.sum(separated_norm * target_norm, dim=-1, keepdim=True) * target_norm

        # Calculate e_noise
        e_noise = separated_norm - s_target

        # Calculate SI-SNR
        si_snr = 20 * torch.log10(
            torch.norm(s_target, dim=-1) / (torch.norm(e_noise, dim=-1) + 1e-8)
        )

        return si_snr.mean()

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch, batch_idx, 'train')
        self.train_step_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch, batch_idx, 'val')
        self.validation_step_outputs.append(metrics)
        return loss

    def on_train_epoch_end(self):
        if self.train_step_outputs:
            avg_metrics = {
                key: np.mean([x[key] for x in self.train_step_outputs])
                for key in self.train_step_outputs[0].keys()
            }
            self.log_dict({f"train_epoch_{k}": v for k, v in avg_metrics.items()})
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_metrics = {
                key: np.mean([x[key] for x in self.validation_step_outputs])
                for key in self.validation_step_outputs[0].keys()
            }
            self.log_dict({f"val_epoch_{k}": v for k, v in avg_metrics.items()})
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=True
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
            }
        }
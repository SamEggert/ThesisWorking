# speaker_separation.py

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav
from models.resunet import ResUNet30
import numpy as np

class SpeakerEncoder:
    def __init__(self):
        self.encoder = VoiceEncoder()

    def get_embedding(self, wav):
        """Get speaker embedding from waveform"""
        if isinstance(wav, str):
            wav = preprocess_wav(wav)
        embed = self.encoder.embed_utterance(wav)
        return torch.from_numpy(embed).float()

class SpeakerSeparation(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        # Initialize separation model
        self.ss_model = ResUNet30(
            input_channels=1,  # Mono audio input
            output_channels=1, # Mono audio output
            condition_size=256 # Size of speaker embeddings
        )

        # Initialize speaker encoder
        self.speaker_encoder = SpeakerEncoder()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.ss_model(x)

    def training_step(self, batch_data_dict, batch_idx):
        """
        Expected batch_data_dict format:
        {
            'mixture': tensor of shape (batch_size, samples),
            'target': tensor of shape (batch_size, samples),
            'speaker_wav': tensor of shape (batch_size, samples) or list of paths
        }
        """
        # Get speaker embeddings
        if isinstance(batch_data_dict['speaker_wav'][0], str):
            # If paths are provided
            embeddings = torch.stack([
                self.speaker_encoder.get_embedding(wav)
                for wav in batch_data_dict['speaker_wav']
            ]).to(self.device)
        else:
            # If waveforms are provided
            embeddings = torch.stack([
                self.speaker_encoder.get_embedding(wav.cpu().numpy())
                for wav in batch_data_dict['speaker_wav']
            ]).to(self.device)

        # Prepare input dictionary
        input_dict = {
            'mixture': batch_data_dict['mixture'].unsqueeze(1),  # Add channel dimension
            'condition': embeddings,
        }

        # Forward pass
        self.ss_model.train()
        output_dict = self.ss_model(input_dict)
        separated = output_dict['waveform'].squeeze(1)  # Remove channel dimension

        # Calculate loss
        loss = torch.nn.functional.l1_loss(separated, batch_data_dict['target'])

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )
        return optimizer

    @torch.no_grad()
    def separate(self, mixture, speaker_wav):
        """
        Separate a target speaker from a mixture.

        Args:
            mixture: torch.Tensor of shape (samples,) or path to wav
            speaker_wav: torch.Tensor of shape (samples,) or path to wav

        Returns:
            torch.Tensor of shape (samples,)
        """
        self.eval()

        # Handle inputs
        if isinstance(mixture, str):
            mixture = preprocess_wav(mixture)
        mixture = torch.from_numpy(mixture).float().to(self.device)

        # Get speaker embedding
        embedding = self.speaker_encoder.get_embedding(speaker_wav).to(self.device)

        # Prepare input
        input_dict = {
            'mixture': mixture.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            'condition': embedding.unsqueeze(0)  # Add batch dimension
        }

        # Separate
        output_dict = self.ss_model(input_dict)
        separated = output_dict['waveform'].squeeze()  # Remove batch and channel dimensions

        return separated
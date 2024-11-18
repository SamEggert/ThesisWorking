import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from speaker_separation import SpeakerSeparation
from resemblyzer.audio import preprocess_wav
import random
import glob
import numpy as np

class VoiceSeparationDataset(Dataset):
    def __init__(
        self,
        target_voices,    # List of paths to sam's voice files
        noise_voices,     # List of paths to bea's voice files
        sequence_length=48000,  # 3 seconds at 16kHz
        min_snr=0,       # Minimum signal-to-noise ratio in dB
        max_snr=5        # Maximum signal-to-noise ratio in dB
    ):
        self.target_voices = target_voices
        self.noise_voices = noise_voices
        self.sequence_length = sequence_length
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __len__(self):
        return len(self.target_voices)

    def __getitem__(self, idx):
        # Load target voice
        target_path = self.target_voices[idx]
        target_wav = preprocess_wav(target_path)

        # Load random noise voice
        noise_path = random.choice(self.noise_voices)
        noise_wav = preprocess_wav(noise_path)

        # Take random chunks if longer than sequence_length
        if len(target_wav) > self.sequence_length:
            start = random.randint(0, len(target_wav) - self.sequence_length)
            target_wav = target_wav[start:start + self.sequence_length]
        else:
            # Pad if too short
            target_wav = np.pad(target_wav, (0, self.sequence_length - len(target_wav)))

        if len(noise_wav) > self.sequence_length:
            start = random.randint(0, len(noise_wav) - self.sequence_length)
            noise_wav = noise_wav[start:start + self.sequence_length]
        else:
            noise_wav = np.pad(noise_wav, (0, self.sequence_length - len(noise_wav)))

        # Mix with random SNR
        target_rms = np.sqrt(np.mean(target_wav**2))
        noise_rms = np.sqrt(np.mean(noise_wav**2))
        snr = random.uniform(self.min_snr, self.max_snr)
        noise_gain = target_rms / (noise_rms * 10**(snr/20))
        mixture = target_wav + noise_gain * noise_wav

        # Normalize
        mixture = mixture / np.max(np.abs(mixture))
        target_wav = target_wav / np.max(np.abs(target_wav))

        return {
            'mixture': torch.from_numpy(mixture).float(),
            'target': torch.from_numpy(target_wav).float(),
            'speaker_wav': target_path  # Original path for speaker embedding
        }

def train_model(
    train_batch_size=8,
    val_batch_size=8,
    num_epochs=100,
    num_workers=4,
    learning_rate=1e-4,
    sequence_length=48000,  # 3 seconds at 16kHz
    checkpoint_dir="checkpoints",
    log_dir="logs",
    gpu_id=0 if torch.cuda.is_available() else None
):
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Get file paths
    sam_voices = glob.glob('audio/sam_Neutral/*.wav')
    bea_voices = glob.glob('audio/bea_Neutral/*.wav')

    # Split into train/val
    random.seed(42)  # For reproducibility
    random.shuffle(sam_voices)
    random.shuffle(bea_voices)


    # Calculate train_size based on the smaller dataset
    train_size = int(0.8 * min(len(sam_voices), len(bea_voices)))

    # Adjust the split for both datasets
    train_target = sam_voices[:train_size]
    val_target = sam_voices[train_size:train_size + int(0.2 * len(sam_voices))]
    train_noise = bea_voices[:train_size]
    val_noise = bea_voices[train_size:train_size + int(0.2 * len(bea_voices))]

    print("\nDataset splits:")
    print(f"Total sam voices: {len(sam_voices)}")
    print(f"Total bea voices: {len(bea_voices)}")
    print(f"Train size: {train_size}")
    print(f"Train target: {len(train_target)}")
    print(f"Val target: {len(val_target)}")
    print(f"Train noise: {len(train_noise)}")
    print(f"Val noise: {len(val_noise)}")

    print(f"Training on {len(train_target)} samples")
    print(f"Validating on {len(val_target)} samples")


    # Create datasets and dataloaders
    train_dataset = VoiceSeparationDataset(
        train_target,
        train_noise,
        sequence_length=sequence_length
    )
    val_dataset = VoiceSeparationDataset(
        val_target,
        val_noise,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # Add this line
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # Add this line
    )

    # Initialize model
    model = SpeakerSeparation(learning_rate=learning_rate)

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_epoch_l1_loss:.3f}',
        save_top_k=3,
        monitor='val_epoch_l1_loss',  # Change this to match the metric you're logging
        mode='min',
        save_last=True
    )

    # Set up early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_epoch_l1_loss',  # Change this to match the metric you're logging
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )

    # Set up logging
    logger = TensorBoardLogger(log_dir, name="speaker_separation")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if gpu_id is not None else 'cpu',
        devices=[gpu_id] if gpu_id is not None else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5      # Validate twice per epoch
    )


    print("trainer:", trainer)
    print("train loader:", train_loader)
    print("val loader:", val_loader)


    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    train_model(
        train_batch_size=8,
        val_batch_size=8,
        num_epochs=100,
        learning_rate=1e-4,
        sequence_length=48000,  # 3 seconds
        checkpoint_dir="checkpoints",
        log_dir="logs",
        gpu_id=0  # Set to None for CPU
    )
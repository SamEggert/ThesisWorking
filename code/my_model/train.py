import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from speaker_separation import SpeakerSeparation
import numpy as np
import random
import glob
from pathlib import Path
import pickle
from tqdm import tqdm

class CachedVCTKDataset(Dataset):
    def __init__(
        self,
        vctk_dir,
        sequence_length=32000,  # Reduced from 48000
        min_snr=0,
        max_snr=5,
        split='train',
        train_ratio=0.8,
        max_files_per_speaker=20,  # Reduced from 50
        max_speakers=20,           # New parameter
        cache_dir='dataset_cache'
    ):
        self.sequence_length = sequence_length
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        cache_file = self.cache_dir / f'vctk_{split}_cache_{max_speakers}spk_{max_files_per_speaker}files.pkl'

        if cache_file.exists():
            print(f"Loading cached {split} dataset...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.preprocessed_data = cache_data['preprocessed_data']
                self.speakers = cache_data['speakers']
                self.speaker_files = cache_data['speaker_files']
        else:
            print(f"Creating new {split} dataset cache...")
            self._create_and_cache_dataset(
                vctk_dir,
                split,
                train_ratio,
                max_files_per_speaker,
                max_speakers,
                cache_file
            )

    def _create_and_cache_dataset(self, vctk_dir, split, train_ratio, max_files_per_speaker, max_speakers, cache_file):
        # Find all speaker directories
        wav_dir = os.path.join(vctk_dir, 'wav48_silence_trimmed')
        speaker_dirs = [d for d in os.listdir(wav_dir)
                       if os.path.isdir(os.path.join(wav_dir, d)) and d.startswith('p')]

        # Limit number of speakers
        random.seed(42)
        random.shuffle(speaker_dirs)
        speaker_dirs = speaker_dirs[:max_speakers]

        # Create dictionary of speaker files
        self.speaker_files = {}
        for speaker in speaker_dirs:
            speaker_path = os.path.join(wav_dir, speaker)
            flac_files = glob.glob(os.path.join(speaker_path, '*.flac'))
            if flac_files:
                # Limit files per speaker
                self.speaker_files[speaker] = flac_files[:max_files_per_speaker]

        # Split speakers
        random.seed(42)
        speakers = list(self.speaker_files.keys())
        random.shuffle(speakers)
        split_idx = int(len(speakers) * train_ratio)

        if split == 'train':
            self.speakers = speakers[:split_idx]
        else:
            self.speakers = speakers[split_idx:]

        # Preprocess and cache all files
        self.preprocessed_data = []
        print("Preprocessing audio files...")
        for speaker in tqdm(self.speakers):
            for file_path in self.speaker_files[speaker]:
                try:
                    processed = self._preprocess_file(file_path)
                    if processed is not None:
                        self.preprocessed_data.append({
                            'audio': processed,
                            'speaker': speaker,
                            'path': file_path
                        })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # Save cache
        cache_data = {
            'preprocessed_data': self.preprocessed_data,
            'speakers': self.speakers,
            'speaker_files': self.speaker_files
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def _preprocess_file(self, file_path):
        # Load and preprocess audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Remove the resampling step to keep original 48kHz
        return waveform.squeeze().numpy()

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        # Get target audio
        target_data = self.preprocessed_data[idx]
        target_wav = target_data['audio']
        target_speaker = target_data['speaker']

        # Get random different speaker
        available_speakers = [s for s in self.speakers if s != target_speaker]
        noise_speaker = random.choice(available_speakers)
        noise_data = random.choice([d for d in self.preprocessed_data if d['speaker'] == noise_speaker])
        noise_wav = noise_data['audio']

        # Handle sequence length
        if len(target_wav) > self.sequence_length:
            start = random.randint(0, len(target_wav) - self.sequence_length)
            target_wav = target_wav[start:start + self.sequence_length]
        else:
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
            'speaker_wav': target_data['path']
        }

def train_model(
    vctk_dir,
    train_batch_size=32,
    val_batch_size=32,
    num_epochs=200,
    num_workers=2,
    learning_rate=3e-4,
    sequence_length=32000,
    checkpoint_dir="checkpoints",
    log_dir="logs",
    max_files_per_speaker=50,
    max_speakers=10,
    resume_from_checkpoint=None
):
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create datasets
    train_dataset = CachedVCTKDataset(
        vctk_dir=vctk_dir,
        sequence_length=sequence_length,
        split='train',
        max_files_per_speaker=max_files_per_speaker,
        max_speakers=max_speakers
    )

    val_dataset = CachedVCTKDataset(
        vctk_dir=vctk_dir,
        sequence_length=sequence_length,
        split='val',
        max_files_per_speaker=max_files_per_speaker,
        max_speakers=max_speakers
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True
    )

    # Initialize model
    if resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {resume_from_checkpoint}")
        model = SpeakerSeparation.load_from_checkpoint(
            resume_from_checkpoint,
            learning_rate=learning_rate
        )
    else:
        print("\nStarting new training...")
        model = SpeakerSeparation(learning_rate=learning_rate)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_epoch_l1_loss:.3f}',
        save_top_k=2,
        monitor='val_epoch_l1_loss',
        mode='min',
        save_last=True,
        every_n_epochs=2,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_epoch_l1_loss',
        min_delta=0.001,
        patience=20,
        verbose=True,
        mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu',
        devices=1,
        strategy='auto',
        precision='16-mixed',
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='{epoch}-{val_epoch_l1_loss:.3f}',
                save_top_k=2,
                monitor='val_epoch_l1_loss',
                mode='min',
                save_last=True,
                every_n_epochs=2,
            ),
            EarlyStopping(
                monitor='val_epoch_l1_loss',
                min_delta=0.001,
                patience=20,
                verbose=True,
                mode='min'
            )
        ],
        logger=TensorBoardLogger(log_dir, name="speaker_separation"),
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint
    )

if __name__ == "__main__":
    vctk_dir = "/scratch/network/se2375/ThesisWorking/code/my_model/audio/VCTK-Corpus-0.92"
    checkpoint_path = None

    train_model(
        vctk_dir=vctk_dir,
        train_batch_size=16,
        val_batch_size=16,
        num_workers=8,
        num_epochs=200,
        learning_rate=3e-4,
        sequence_length=192000,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        max_files_per_speaker=172,
        max_speakers=109,
        resume_from_checkpoint=checkpoint_path
    )

import torch
import torchaudio
import numpy as np
from speaker_separation import SpeakerSeparation
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import random
import glob
import lightning.pytorch as pl

def plot_waveforms(mixture, separated, target, output_dir):
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(mixture)
    plt.title('Mixture')
    plt.ylabel('Amplitude')
    plt.subplot(3, 1, 2)
    plt.plot(separated)
    plt.title('Separated')
    plt.ylabel('Amplitude')
    plt.subplot(3, 1, 3)
    plt.plot(target)
    plt.title('Target')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveform_comparison.png'))
    plt.close()

def calculate_metrics(separated, target, mixture):
    separated = torch.from_numpy(separated)
    target = torch.from_numpy(target)
    mixture = torch.from_numpy(mixture)

    noise = separated - target
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

    orig_noise = mixture - target
    orig_noise_power = torch.mean(orig_noise ** 2)
    orig_snr = 10 * torch.log10(signal_power / (orig_noise_power + 1e-10))

    separated_norm = separated - torch.mean(separated)
    target_norm = target - torch.mean(target)

    separated_norm = separated_norm / torch.norm(separated_norm)
    target_norm = target_norm / torch.norm(target_norm)

    s_target = torch.sum(separated_norm * target_norm) * target_norm
    e_noise = separated_norm - s_target

    si_snr = 20 * torch.log10(torch.norm(s_target) / (torch.norm(e_noise) + 1e-8))

    return {
        'snr': snr.item(),
        'original_snr': orig_snr.item(),
        'si_snr': si_snr.item(),
        'difference': torch.mean(torch.abs(separated - mixture)).item()
    }

def load_and_preprocess_wav(file_path):
    # Load original audio at 48kHz
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Get 48kHz version for separation
    separation_wav = waveform.squeeze().numpy()

    # Create resampler for 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    # Resample to 16kHz for resemblyzer
    wav_16k = resampler(waveform)
    embedding_wav = preprocess_wav(wav_16k.squeeze().numpy(), source_sr=16000)

    return separation_wav, embedding_wav

def get_random_speaker_file(vctk_dir, exclude_speaker=None):
    wav_dir = os.path.join(vctk_dir, 'wav48_silence_trimmed')
    speaker_dirs = [d for d in os.listdir(wav_dir)
                   if os.path.isdir(os.path.join(wav_dir, d)) and d.startswith('p')]

    if exclude_speaker:
        speaker_dirs.remove(exclude_speaker)

    speaker = random.choice(speaker_dirs)
    speaker_path = os.path.join(wav_dir, speaker)
    flac_files = glob.glob(os.path.join(speaker_path, '*.flac'))
    return random.choice(flac_files), speaker

def create_mixed_signal(target_wav, noise_wav):
    min_length = min(len(target_wav), len(noise_wav))
    target_wav = target_wav[:min_length]
    noise_wav = noise_wav[:min_length]

    target_rms = np.sqrt(np.mean(target_wav**2))
    noise_rms = np.sqrt(np.mean(noise_wav**2))
    noise_gain = target_rms / noise_rms
    mixture = target_wav + noise_gain * noise_wav

    mixture = mixture / np.max(np.abs(mixture))
    target_wav = target_wav / np.max(np.abs(target_wav))

    return mixture, target_wav

def main():
    # Define paths
    VCTK_DIR = "/scratch/network/se2375/ThesisWorking/code/my_model/audio/VCTK-Corpus-0.92"
    CHECKPOINT_PATH = "checkpoints/epoch=9-val_epoch_l1_loss=0.020.ckpt"
    OUTPUT_DIR = "test_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize PyTorch Lightning trainer with GPU
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision='16-mixed'
    )

    print("Selecting random speakers...")
    target_path, target_speaker = get_random_speaker_file(VCTK_DIR)
    noise_path, noise_speaker = get_random_speaker_file(VCTK_DIR, exclude_speaker=target_speaker)

    print(f"Target speaker: {target_speaker}")
    print(f"Noise speaker: {noise_speaker}")

    print("Loading audio files...")
    # Load both 48kHz and 16kHz versions for target speaker
    target_wav_48k, target_wav_16k = load_and_preprocess_wav(target_path)
    # Load only 48kHz version for noise speaker
    noise_wav_48k, _ = load_and_preprocess_wav(noise_path)

    print("Creating mixed signal...")
    mixture, target_wav = create_mixed_signal(target_wav_48k, noise_wav_48k)

    print("Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    state_dict = {k: v for k, v in checkpoint['state_dict'].items()
                if not k.startswith('speaker_encoder.')}
    checkpoint['state_dict'] = state_dict
    model = SpeakerSeparation.load_from_checkpoint(CHECKPOINT_PATH, state_dict=state_dict)
    model = model.cuda().eval()

    print("Computing speaker embedding...")
    encoder = VoiceEncoder().cuda()
    with torch.no_grad():
        embedding = encoder.embed_utterance(target_wav_16k)
    speaker_embedding = torch.from_numpy(embedding).float().cuda()

    print("Preparing input tensors...")
    mixture_tensor = torch.from_numpy(mixture).float().cuda()
    mixture_tensor = mixture_tensor.view(1, 1, -1)

    print("Running separation...")
    with torch.no_grad():
        output_dict = model.ss_model({
            'mixture': mixture_tensor,
            'condition': speaker_embedding.unsqueeze(0),
        })
        separated_wav = output_dict['waveform'].squeeze().cpu().numpy()

    print("Calculating metrics...")
    metrics = calculate_metrics(separated_wav, target_wav, mixture)
    print("\nSeparation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    print("Generating visualizations...")
    plot_waveforms(mixture, separated_wav, target_wav, OUTPUT_DIR)

    print("\nSaving audio files...")
    wavfile.write(os.path.join(OUTPUT_DIR, "mixture.wav"), 48000, mixture.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "separated.wav"), 48000, separated_wav.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "original.wav"), 48000, target_wav.astype(np.float32))

    print("Saving test information...")
    with open(os.path.join(OUTPUT_DIR, "test_info.txt"), "w") as f:
        f.write(f"Target Speaker: {target_speaker}\n")
        f.write(f"Target File: {target_path}\n")
        f.write(f"Noise Speaker: {noise_speaker}\n")
        f.write(f"Noise File: {noise_path}\n")
        f.write("\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")

    print("Processing complete!")

if __name__ == "__main__":
    main()
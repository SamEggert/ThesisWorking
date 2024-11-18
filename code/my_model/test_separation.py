import torch
import torchaudio
import numpy as np
from speaker_separation import SpeakerSeparation
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

def plot_waveforms(mixture, separated, target, output_dir):
    """Plot waveforms for visual comparison"""
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
    """Calculate separation metrics"""
    # Convert to tensor
    separated = torch.from_numpy(separated)
    target = torch.from_numpy(target)
    mixture = torch.from_numpy(mixture)

    # SNR
    noise = separated - target
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

    # Original mixture SNR
    orig_noise = mixture - target
    orig_noise_power = torch.mean(orig_noise ** 2)
    orig_snr = 10 * torch.log10(signal_power / (orig_noise_power + 1e-10))

    # SI-SNR
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

def separate_voice(model, mixture, speaker_wav, device='cpu'):
    """Separate voice using the trained model"""
    # Convert mixture to tensor and reshape
    mixture_tensor = torch.from_numpy(mixture).float().to(device)
    mixture_tensor = mixture_tensor.view(1, 1, -1)  # [1, 1, samples]

    # Get speaker embedding
    wav = preprocess_wav(speaker_wav)
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)
    speaker_embedding = torch.from_numpy(embedding).float().to(device)

    # Create input dict
    input_dict = {
        'mixture': mixture_tensor,
        'condition': speaker_embedding.unsqueeze(0)
    }

    # Print shapes for debugging
    print(f"\nInput shapes:")
    print(f"Mixture tensor shape: {mixture_tensor.shape}")
    print(f"Speaker embedding shape: {speaker_embedding.shape}")

    # Run inference
    with torch.no_grad():
        output_dict = model(input_dict)
        separated = output_dict['waveform'].squeeze().cpu().numpy()

    print(f"Output shape: {separated.shape}")

    return separated

def create_mixed_signal(target_wav, noise_wav):
    """Create a mixture signal, handling different lengths"""
    # Print original lengths
    print(f"Original lengths - Target: {len(target_wav)}, Noise: {len(noise_wav)}")

    # Use shorter length
    min_length = min(len(target_wav), len(noise_wav))

    # Truncate both to minimum length
    target_wav = target_wav[:min_length]
    noise_wav = noise_wav[:min_length]

    print(f"After truncation - Target: {len(target_wav)}, Noise: {len(noise_wav)}")

    # Mix with 0dB SNR
    target_rms = np.sqrt(np.mean(target_wav**2))
    noise_rms = np.sqrt(np.mean(noise_wav**2))
    noise_gain = target_rms / noise_rms
    mixture = target_wav + noise_gain * noise_wav

    # Normalize
    mixture = mixture / np.max(np.abs(mixture))
    target_wav = target_wav / np.max(np.abs(target_wav))

    return mixture, target_wav

def main():
    # Define paths
    BASE_DIR = "/Users/samueleggert/GitHub/ThesisWorking/code/my_model"
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints/last-v2.ckpt") # checkpoints are variable to change
    SAM_WAV = os.path.join(BASE_DIR, "audio/sam_Neutral/neutral_1-28_0001.wav")
    BEA_WAV = os.path.join(BASE_DIR, "audio/bea_Neutral/Neutral_1-28_0001.wav")
    OUTPUT_DIR = os.path.join(BASE_DIR, "test_outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and preprocess the wav files
    print("Loading audio files...")
    target_wav = preprocess_wav(SAM_WAV)
    noise_wav = preprocess_wav(BEA_WAV)

    # Create mixture
    print("Creating mixed signal...")
    mixture, target_wav = create_mixed_signal(target_wav, noise_wav)

    # Load model
    print("Loading model...")
    model = SpeakerSeparation.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    # Separate
    print("Performing separation...")
    separated_wav = separate_voice(model, mixture, SAM_WAV, device)

    # Calculate metrics
    metrics = calculate_metrics(separated_wav, target_wav, mixture)
    print("\nSeparation Metrics:")
    print(f"SNR: {metrics['snr']:.2f} dB")
    print(f"Original Mixture SNR: {metrics['original_snr']:.2f} dB")
    print(f"SI-SNR: {metrics['si_snr']:.2f} dB")
    print(f"Mean absolute difference from mixture: {metrics['difference']:.4f}")

    # Plot waveforms
    plot_waveforms(mixture, separated_wav, target_wav, OUTPUT_DIR)

    # Save audio files
    print("\nSaving results...")
    wavfile.write(os.path.join(OUTPUT_DIR, "mixture.wav"), 16000, mixture.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "separated.wav"), 16000, separated_wav.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "original.wav"), 16000, target_wav.astype(np.float32))

    print(f"\nDone! Check the output files in {OUTPUT_DIR}")
    print("- mixture.wav: The mixed signal (Sam + Bea)")
    print("- separated.wav: The separated voice (should be Sam)")
    print("- original.wav: The original clean voice (Sam)")
    print("- waveform_comparison.png: Visual comparison of the waveforms")

if __name__ == "__main__":
    main()
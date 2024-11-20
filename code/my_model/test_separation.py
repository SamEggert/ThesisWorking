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
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    waveform = waveform.squeeze().numpy()
    waveform = preprocess_wav(waveform, source_sr=16000)

    return waveform

def separate_voice(model, mixture, speaker_wav, device='cpu'):
    mixture_tensor = torch.from_numpy(mixture).float().to(device)
    mixture_tensor = mixture_tensor.view(1, 1, -1)

    wav = load_and_preprocess_wav(speaker_wav)
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)
    speaker_embedding = torch.from_numpy(embedding).float().to(device)

    input_dict = {
        'mixture': mixture_tensor,
        'condition': speaker_embedding.unsqueeze(0)
    }

    with torch.no_grad():
        output_dict = model(input_dict)
        separated = output_dict['waveform'].squeeze().cpu().numpy()

    return separated

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
    BASE_DIR = "/Users/samueleggert/GitHub/ThesisWorking"
    TARGET_PATH = os.path.join(BASE_DIR, "audio/bea_Neutral/Neutral_1-28_0001.wav")
    NOISE_PATH = os.path.join(BASE_DIR, "audio/sam_Neutral/neutral_1-28_0001.wav")
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "code/my_model/checkpoints/epoch=42-val_epoch_l1_loss=0.032.ckpt")
    OUTPUT_DIR = os.path.join(BASE_DIR, "code/my_model/test_outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cpu'

    print("Loading audio files...")
    target_wav = load_and_preprocess_wav(TARGET_PATH)
    noise_wav = load_and_preprocess_wav(NOISE_PATH)

    print("Creating mixed signal...")
    mixture, target_wav = create_mixed_signal(target_wav, noise_wav)

    print("Loading model...")
    model = SpeakerSeparation.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    print("Performing separation...")
    separated_wav = separate_voice(model, mixture, TARGET_PATH, device)

    metrics = calculate_metrics(separated_wav, target_wav, mixture)
    print("\nSeparation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    plot_waveforms(mixture, separated_wav, target_wav, OUTPUT_DIR)

    print("\nSaving audio files...")
    wavfile.write(os.path.join(OUTPUT_DIR, "mixture.wav"), 16000, mixture.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "separated.wav"), 16000, separated_wav.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "original.wav"), 16000, target_wav.astype(np.float32))

    with open(os.path.join(OUTPUT_DIR, "test_info.txt"), "w") as f:
        f.write(f"Target Speaker File: {TARGET_PATH}\n")
        f.write(f"Noise Speaker File: {NOISE_PATH}\n")
        f.write("\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")

if __name__ == "__main__":
    main()
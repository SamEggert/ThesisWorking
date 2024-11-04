import torch
import torchaudio
import numpy as np
from speaker_separation import SpeakerSeparation
from resemblyzer.audio import preprocess_wav
from scipy.io import wavfile

def create_mixed_signal(target_wav, noise_wav=None):
    """Create a mixture by adding noise or creating synthetic interference"""
    # If no noise provided, create synthetic noise
    if noise_wav is None:
        # Create synthetic noise (sine wave at different frequencies)
        t = np.linspace(0, len(target_wav) / 16000, len(target_wav))
        noise = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        noise = noise.astype(np.float32)
    else:
        noise = noise_wav
        # Make sure noise is same length as target
        if len(noise) > len(target_wav):
            noise = noise[:len(target_wav)]
        else:
            noise = np.pad(noise, (0, len(target_wav) - len(noise)))

    # Mix the signals
    mixture = target_wav + noise

    # Normalize
    mixture = mixture / np.max(np.abs(mixture))

    return mixture

def main():
    # Load and preprocess the example wav file
    example_wav_path = "neutral_1-28_0001.wav"
    target_wav = preprocess_wav(example_wav_path)

    # Create a mixture
    mixture = create_mixed_signal(target_wav)

    # Initialize model
    model = SpeakerSeparation()

    print("Attempting separation...")

    # Perform separation
    separated_wav = model.separate(
        mixture=mixture,
        speaker_wav=example_wav_path
    )

    # Convert to numpy
    separated_wav = separated_wav.cpu().numpy()

    # Save results
    print("Saving results...")
    wavfile.write("mixture.wav", 16000, mixture.astype(np.float32))
    wavfile.write("separated.wav", 16000, separated_wav.astype(np.float32))
    wavfile.write("original.wav", 16000, target_wav.astype(np.float32))

    print("Done! Check the output files:")
    print("- mixture.wav: The mixed input signal")
    print("- separated.wav: The separated voice")
    print("- original.wav: The original clean voice")

if __name__ == "__main__":
    main()
import torch
import torchaudio
import numpy as np
from speaker_separation import SpeakerSeparation
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav
from scipy.io import wavfile
import os

def create_mixed_signal(target_wav, noise_wav, snr_db=0):
    """Create a mixture of target and noise voices with specified SNR"""
    # Make sure noise is same length as target
    if len(noise_wav) > len(target_wav):
        noise_wav = noise_wav[:len(target_wav)]
    else:
        noise_wav = np.pad(noise_wav, (0, len(target_wav) - len(noise_wav)))

    # Calculate scaling factor for desired SNR
    target_rms = np.sqrt(np.mean(target_wav**2))
    noise_rms = np.sqrt(np.mean(noise_wav**2))
    scaling = target_rms / (noise_rms * 10**(snr_db/20))

    # Mix the signals
    mixture = target_wav + scaling * noise_wav

    # Normalize
    mixture = mixture / np.max(np.abs(mixture))

    return mixture

def get_speaker_embedding(wav_path):
    """Get speaker embedding using Resemblyzer"""
    # Load and preprocess audio
    wav = preprocess_wav(wav_path)

    # Initialize encoder
    encoder = VoiceEncoder()

    # Get embedding
    embedding = encoder.embed_utterance(wav)
    return torch.from_numpy(embedding).float()

def separate_voice(model, mixture, speaker_wav, device='cpu'):
    """Separate voice using the trained model"""
    # Convert mixture to tensor and reshape
    # Model expects: [batch_size, channels, samples]
    mixture_tensor = torch.from_numpy(mixture).float().to(device)
    mixture_tensor = mixture_tensor.view(1, 1, -1)  # [1, 1, samples]

    # Get speaker embedding
    speaker_embedding = get_speaker_embedding(speaker_wav).to(device)

    # Create input dict
    input_dict = {
        'mixture': mixture_tensor,  # Shape: [1, 1, samples]
        'condition': speaker_embedding.unsqueeze(0)  # Shape: [1, embedding_size]
    }

    # Run inference
    with torch.no_grad():
        output_dict = model(input_dict)
        separated = output_dict['waveform'].squeeze().cpu().numpy()

    return separated

def main():
    # Define paths
    BASE_DIR = "/Users/samueleggert/GitHub/ThesisWorking/code/my_model"
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints/last.ckpt")
    SAM_WAV = os.path.join(BASE_DIR, "audio/sam_Neutral/neutral_1-28_0001.wav")
    BEA_WAV = os.path.join(BASE_DIR, "audio/bea_Neutral/Neutral_1-28_0001.wav")
    OUTPUT_DIR = os.path.join(BASE_DIR, "test_outputs")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and preprocess the wav files
    print("Loading audio files...")
    target_wav = preprocess_wav(SAM_WAV)
    noise_wav = preprocess_wav(BEA_WAV)

    # Print some debug information
    print(f"Target wav shape: {target_wav.shape}")
    print(f"Noise wav shape: {noise_wav.shape}")

    # Create mixture
    print("Creating mixed signal...")
    mixture = create_mixed_signal(target_wav, noise_wav, snr_db=0)
    print(f"Mixture shape: {mixture.shape}")

    # Initialize model and load checkpoint
    print("Loading model from checkpoint...")
    model = SpeakerSeparation.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    print("Performing separation...")
    separated_wav = separate_voice(model, mixture, SAM_WAV, device)
    print(f"Separated wav shape: {separated_wav.shape}")

    # Save results
    print("Saving results...")
    wavfile.write(os.path.join(OUTPUT_DIR, "mixture.wav"), 16000, mixture.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "separated.wav"), 16000, separated_wav.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "original.wav"), 16000, target_wav.astype(np.float32))
    wavfile.write(os.path.join(OUTPUT_DIR, "noise.wav"), 16000, noise_wav.astype(np.float32))

    print(f"\nDone! Check the output files in {OUTPUT_DIR}:")
    print("- mixture.wav: The mixed signal (Sam + Bea)")
    print("- separated.wav: The separated voice (should be Sam)")
    print("- original.wav: The original clean voice (Sam)")
    print("- noise.wav: The noise voice (Bea)")

if __name__ == "__main__":
    main()
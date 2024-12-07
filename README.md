# Speaker Separation Model

This repository contains an implementation of a speaker separation model using a ResUNet architecture with Feature-wise Linear Modulation (FiLM). The model is designed to separate mixed speech signals by targeting a specific speaker's voice using their speaker embedding as a condition.

## Project Structure

The main implementation is located at:
```
/ThesisWorking/code/my_model/
```

Key files:
- `speaker_separation.py`: Main model implementation using PyTorch Lightning
- `models/resunet.py`: ResUNet architecture with FiLM conditioning
- `train.py`: Training pipeline with VCTK dataset support
- `test_separation.py`: Evaluation script for the trained model

## Core Components

1. **Speaker Separation Model** (`speaker_separation.py`)
   - PyTorch Lightning implementation
   - Uses speaker embeddings for targeted separation
   - Includes multiple evaluation metrics (SNR, SI-SNR)

2. **ResUNet Architecture** (`models/resunet.py`)
   - Modified ResUNet with FiLM conditioning
   - Handles spectrogram processing and waveform reconstruction
   - Implements skip connections and residual blocks

3. **Training Pipeline** (`train.py`)
   - Supports the VCTK dataset
   - Implements data caching for faster training
   - Includes validation and checkpoint saving

## Getting Started

1. Set up the environment:
```bash
# Execute the setup script
bash setup_training.sh
```

2. Train the model:
```bash
# For full training
python train.py

# For a quick test run
python train.py --test_mode
```

3. Test the model:
```bash
# Run inference on test samples
python test_separation.py
```

## Hardware Requirements

- GPU with CUDA support (tested on A100)
- Minimum 32GB system memory
- Recommended: 8 CPU cores

## Additional Notes

- The model uses the VCTK dataset for training
- Checkpoints are saved in the `checkpoints` directory
- Training logs can be monitored using TensorBoard

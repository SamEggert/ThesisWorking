#!/bin/bash
#SBATCH --job-name=sep_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=se2375@princeton.edu

module purge
module load anaconda3/2023.3

# Get the absolute path to the current directory
CURRENT_DIR=$(pwd)

# Activate virtual environment using absolute path
source ${CURRENT_DIR}/venv/bin/activate

# Create temporary test config
cat > ${CURRENT_DIR}/test_config.py << EOF
# Override training parameters for quick test
TEST_CONFIG = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'num_epochs': 2,
    'max_files_per_speaker': 10,
    'max_speakers': 5
}
EOF

# Run the training script with test configuration
python -c "
import train
from test_config import TEST_CONFIG

train.train_model(
    vctk_dir='/scratch/network/se2375/ThesisWorking/code/my_model/audio/VCTK-Corpus-0.92',
    train_batch_size=TEST_CONFIG['train_batch_size'],
    val_batch_size=TEST_CONFIG['val_batch_size'],
    num_epochs=TEST_CONFIG['num_epochs'],
    num_workers=4,
    learning_rate=3e-4,
    sequence_length=32000,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    max_files_per_speaker=TEST_CONFIG['max_files_per_speaker'],
    max_speakers=TEST_CONFIG['max_speakers'],
    resume_from_checkpoint=None
)"

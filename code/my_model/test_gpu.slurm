#!/bin/bash
#SBATCH --job-name=sep_test        # create a short name for your job
#SBATCH --nodes=1                  # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=4         # cpu-cores per task (reduced for test)
#SBATCH --mem=16G                 # total memory per node (reduced for test)
#SBATCH --gres=gpu:1              # number of gpus per node
#SBATCH --constraint=a100         # specify GPU architecture
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS) - 15 minutes
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=se2375@princeton.edu

module purge
module load anaconda3/2023.3

# Activate virtual environment
source /home/$USER/my_model/venv/bin/activate

# Create temporary test config by modifying main script
cat > test_config.py << 'EOL'
# Override training parameters for quick test
TEST_CONFIG = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'num_epochs': 2,
    'max_files_per_speaker': 10,
    'max_speakers': 5
}
EOL

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
)
"
EOL

# Save this as a new script
cat > test_gpu.slurm << 'EOL'
#!/bin/bash
#SBATCH --job-name=sep_test        # create a short name for your job
#SBATCH --nodes=1                  # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=4         # cpu-cores per task (reduced for test)
#SBATCH --mem=16G                 # total memory per node (reduced for test)
#SBATCH --gres=gpu:1              # number of gpus per node
#SBATCH --constraint=a100         # specify GPU architecture
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS) - 15 minutes
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=se2375@princeton.edu

module purge
module load anaconda3/2023.3

# Activate virtual environment
source /home/$USER/my_model/venv/bin/activate

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
#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning
pip install torchaudio
pip install resemblyzer
pip install matplotlib
pip install scipy
pip install torchlibrosa
pip install tqdm

# Create Slurm job submission script
cat > train_gpu.slurm << 'EOL'
#!/bin/bash
#SBATCH --job-name=speaker_sep        # create a short name for your job
#SBATCH --nodes=1                     # node count
#SBATCH --ntasks=1                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8             # cpu-cores per task
#SBATCH --mem=32G                     # total memory per node
#SBATCH --gres=gpu:1                  # number of gpus per node
#SBATCH --constraint=a100             # specify GPU architecture
#SBATCH --time=24:00:00              # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-type=fail             # send mail if job fails
#SBATCH --mail-user=se2375@princeton.edu

module purge
module load anaconda3/2023.3

# Activate virtual environment
source /home/$USER/venv/bin/activate

# Run the training script
python train.py
EOL

chmod +x train_gpu.slurm
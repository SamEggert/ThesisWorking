#!/bin/bash
#SBATCH --job-name=sep_test          # create a short name for your job
#SBATCH --nodes=1                     # node count
#SBATCH --ntasks=1                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8             # cpu-cores per task
#SBATCH --mem=32G                     # total memory per node
#SBATCH --gres=gpu:1                  # number of gpus per node
#SBATCH --constraint=a100             # specify GPU architecture
#SBATCH --time=00:30:00              # 30 minutes
#SBATCH --qos=gpu-short              # use short queue
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-type=fail             # send mail if job fails
#SBATCH --mail-user=se2375@princeton.edu

module purge
module load anaconda3/2023.3

# Activate virtual environment
source /scratch/network/se2375/ThesisWorking/code/my_model/venv/bin/activate

# Run the test script
python test_separation.py
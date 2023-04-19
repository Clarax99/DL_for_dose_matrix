#!/bin/bash
#SBATCH --job-name=4_CNN2feat_noreg_50_weighted_loss
#SBATCH --output=%x.o%j
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --mail-user=clara.cousteix@student-cs.fr
#SBATCH --mail-type=ALL

# Module load
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.7.0/intel-20.0.4.304

# Activate anaconda environment code
source activate pytorch

# Train the network
python main.py 50 -g
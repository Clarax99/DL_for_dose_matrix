#!/bin/bash
#SBATCH --job-name=0_newnet_noreg_40_3feat_noweighted_loss
#SBATCH --output=%x.o%j
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=clara.cousteix@student-cs.fr
#SBATCH --mail-type=ALL

# Module load
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.7.0/intel-20.0.4.304

# Activate anaconda environment code
source activate pytorch

# Train the network
python main.py 40 -g
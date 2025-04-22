#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --job-name=train_rl-agent
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --mem=40G              # 16 gigabytes memory
#SBATCH --cpus-per-task=2     # 2 CPU cores
#SBATCH --output=output.txt    # Log file
#SBATCH --mail-user=henrihl@stud.ntnu.no
#SBATCH --mail-type=ALL

module load Anaconda3/2024.02-1

conda activate rl_env

python ./main.py

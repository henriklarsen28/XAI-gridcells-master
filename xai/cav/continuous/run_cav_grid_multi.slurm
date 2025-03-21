#!/bin/bash

start_grid=$1
end_grid=$2

echo "Running with $start_grid"
output_file='output_${start_grid}.txt'
#SBATCH --partition=CPUQ
#SBATCH --job-name=grid_cav
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:30:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --mem=75G              # 16 gigabytes memory
#SBATCH --cpus-per-task=4     # 2 CPU cores
#SBATCH --output='$output_file'    # Log file
#SBATCH --mail-user=henrihl@stud.ntnu.no
#SBATCH --mail-type=ALL

module load Anaconda3/2024.02-1

conda activate rl_env



python ./grid_cav_idun.py "$start_grid" "$end_grid"

#!/bin/bash

for i in {0..42..7}; do
j=$((i+7))
sbatch run_cav_grid_multi.slurm "$i" "$j"
done

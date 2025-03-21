#!/bin/bash

for i in {0..49..7} do
    sbatch run_cav_grid_multi.slurm $i $(i+7)
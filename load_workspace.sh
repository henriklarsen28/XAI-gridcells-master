#!/bin/bash
CONDA_ENV="xai_master_idun"

cd XAI-gridcells-master
module load Anaconda3/2024.02-1
conda activate $CONDA_ENV
pip install -r requirements.txt
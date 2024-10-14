#!/bin/bash

# To set up file: 
# move this file to the root directory from the XAI-gridcells-master using "mv load.sh .."
# make it executable with "chmod +x load.sh"

CONDA_ENV="xai_master_idun"

cd XAI-gridcells-master
module load Anaconda3/2024.02-1
conda activate $CONDA_ENV
pip install -r requirements.txt
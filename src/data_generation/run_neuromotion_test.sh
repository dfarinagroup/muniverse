#!/bin/bash
set -e  # fail on first error

# Initialize conda
source /opt/conda/etc/profile.d/conda.sh

# Activate NeuroMotion env
conda activate NeuroMotion

# Move to NeuroMotion folder
cd /opt/NeuroMotion

# Run get_data and generate EMG
python scripts/mov2emg.py

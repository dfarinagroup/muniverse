#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from src.algorithms.decomposition import decompose_scd

# TODO: Extend script to handle multiple recordings
# TODO: Handle start and end times; sampling frequency; etc. -- config is modified here and then passed to the function
# TODO: Config is passed as a dict, and then saved as a json file in the temp directory along with the npy data

# Input/output paths
data_path = project_root / 'neuromotion-dev01/sub-sim02/emg/sub-sim02_task-ECRBdynamicradialsinusoid21percentmvcsub2ncol10_run-01_emg.edf'
output_dir = project_root / 'outputs'
container = project_root / 'environment/muniverse_scd.sif'
config = project_root / 'configs/scd.json'

# Run decomposition
decompose_scd(
    data=str(data_path),
    output_dir=str(output_dir),
    algorithm_config=str(config),
    engine='singularity',
    container=str(container)
)
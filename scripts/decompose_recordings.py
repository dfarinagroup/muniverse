#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from pathlib import Path
from muniverse.algorithms.decomposition import decompose_scd

# TODO: Next step - make it a BIDS compatible function

DATASET_NAME = 'neuromotion-test'
CONFIG = f'/rds/general/user/pm1222/home/muniverse-demo/configs/scd.json'
DATA_DIR = f'/rds/general/user/pm1222/ephemeral/muniverse/datasets/outputs/{DATASET_NAME}'
OUTPUT_DIR = f'/rds/general/user/pm1222/ephemeral/muniverse/interim/scd_outputs/{DATASET_NAME}'
CONTAINER = f'/rds/general/user/pm1222/home/muniverse-demo/environment/muniverse_scd.sif'


def process_recording(run_dir: Path, output_dir: Path, algorithm_config: str, container: str):
    """
    Process a single recording directory.
    
    Args:
        run_dir (Path): Path to the recording directory
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        container (str): Path to the Singularity container
    """
    # Find the log file
    log_files = list(run_dir.glob('*_log.json'))
    if not log_files:
        print(f"No log file found in {run_dir}")
        return
    
    log_file = log_files[0]
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Check movement profile
    movement_config = log_data['InputData']['Configuration']['MovementConfiguration']
    movement_type = movement_config['MovementType']
    effort_profile = movement_config['MovementProfileParameters']['EffortProfile']
    if movement_type != 'Isometric' and effort_profile != 'Trapezoid':
        print(f"Skipping {run_dir.name} - not isometric trapezoidal")
        return
    
    # Get rest duration and sampling frequency
    rest_duration = movement_config['MovementProfileParameters']['RestDuration']
    ramp_duration = movement_config['MovementProfileParameters']['RampDuration']
    sampling_frequency = log_data['InputData']['Configuration']['RecordingConfiguration']['SamplingFrequency']
    n_electrodes = log_data['InputData']['Configuration']['RecordingConfiguration']['ElectrodeConfiguration']['NElectrodes']
    
    # Set start and end times
    start_time = rest_duration + ramp_duration
    end_time = -rest_duration - ramp_duration
    
    # Create output directory for this recording
    recording_output_dir = output_dir
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and modify algorithm config
    with open(algorithm_config, 'r') as f:
        algo_config = json.load(f)
    algo_config['Config']['start_time'] = start_time
    algo_config['Config']['end_time'] = end_time
    algo_config['Config']['sampling_frequency'] = sampling_frequency
    algo_config['Config']['extension_factor'] = int(1000 / n_electrodes) + 1
    
    # Save modified config to output directory
    modified_config_path = recording_output_dir / 'algorithm_config.json'
    with open(modified_config_path, 'w') as f:
        json.dump(algo_config, f, indent=2)
    print(f"Saved modified algorithm config to {modified_config_path}")
    
    # Find the EMG file
    emg_files = list(run_dir.glob('*_emg.npz'))
    if not emg_files:
        print(f"No EMG file found in {run_dir}")
        return
    
    emg_file = emg_files[0]
    
    # Convert NPZ to NPY
    try:
        # Load the NPZ file
        npz_data = np.load(emg_file)
        # Get the EMG data (assuming it's stored in the 'emg' key)
        emg_data = npz_data['emg']
        # Create NPY file path
        npy_file = run_dir / f"{emg_file.stem}.npy"
        # Save as NPY
        np.save(npy_file, emg_data)
        print(f"Converted {emg_file.name} to NPY format, saved to {npy_file}")
    except Exception as e:
        print(f"Error converting NPZ to NPY for {run_dir.name}: {str(e)}")
        return
    
    # Run decomposition
    try:
        decompose_scd(
            data=str(npy_file),
            output_dir=str(recording_output_dir),
            algorithm_config=str(modified_config_path),
            engine='singularity',
            container=container
        )
        print(f"Successfully decomposed recording in {run_dir}")
    except Exception as e:
        print(f"Error processing {run_dir.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Decompose EMG recordings using SCD algorithm')
    parser.add_argument('--data_dir', default=DATA_DIR  ,
                      help='Directory containing recording directories')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                      help='Output directory for decomposition results')
    parser.add_argument('--algorithm_config', default=CONFIG,
                      help='Path to algorithm configuration file')
    parser.add_argument('--container', default=CONTAINER,
                      help='Path to Singularity container')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each recording directory
    for run_dir in data_dir.iterdir():
        if run_dir.is_dir():
            print(f"\nProcessing {run_dir.name}...")
            process_recording(run_dir, output_dir, args.algorithm_config, args.container)

if __name__ == '__main__':
    main()
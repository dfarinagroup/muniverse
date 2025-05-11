#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
import edfio
import scipy
from pathlib import Path
from muniverse.algorithms.decomposition import decompose_scd
from typing import Optional
import re


DATASET_NAME = 'Grison_et_al_2025'
BIDS_ROOT = f'/rds/general/user/pm1222/ephemeral/muniverse/datasets/bids/{DATASET_NAME}'
CONFIG = f'/rds/general/user/pm1222/home/muniverse-demo/configs/scd.json'
OUTPUT_DIR = f'/rds/general/user/pm1222/ephemeral/muniverse/interim/scd_outputs/{DATASET_NAME}'
CONTAINER = f'/rds/general/user/pm1222/home/muniverse-demo/environment/muniverse_scd.sif'


def extract_bids_components(filename: str) -> tuple[str, str, str]:
    """Extract BIDS components (subject, session, task) from filename."""
    # Define patterns for each component
    patterns = {
        'subject': r'sub-(\d+)',
        'task': r'task-(\w+)_run'
    }
    
    # Extract each component
    components = {}
    for component, pattern in patterns.items():
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(f"Could not find {component} in filename: {filename}")
        components[component] = match.group(1)
    
    return components['subject'], components['task']


def create_bids_dataframe(bids_root: Path) -> pd.DataFrame:
    """Create a DataFrame containing all EDF files in the BIDS structure and their corresponding simulation log paths."""
    # Find all EDF files recursively
    edf_files = list(bids_root.rglob('*_emg.edf'))
    
    # Create lists to store the data
    data = []
    
    for edf_path in edf_files:
        try:
            # Extract BIDS components
            subject, task = extract_bids_components(edf_path.stem)
            
            # Determine if this is simulated or experimental data
            log_path = edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_simulation.json"
            emg_sidecar = edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_emg.json"
            data_type = 'simulated' if log_path.exists() else 'experimental'
            
            data.append({
                'edf_path': str(edf_path),
                'log_config_path': str(log_path) if data_type == 'simulated' else "n/a",
                'emg_sidecar_path': str(emg_sidecar) if data_type == 'experimental' else "n/a",
                'subject': subject,
                'task': task,
                'data_type': data_type
            })
        except ValueError as e:
            print(f"Warning: Skipping file {edf_path.name} - {str(e)}")
            continue
    
    if not data:
        raise ValueError("No valid EDF files found in the BIDS structure")
    
    return pd.DataFrame(data)


def get_simulation_config(log_config_path: Path) -> dict:
    """Extract configuration from simulation log file.
    
    Args:
        log_config_path (Path): Path to the simulation log file
        
    Returns:
        dict: Configuration dictionary containing:
            - start_time: Start time for analysis
            - end_time: End time for analysis
            - sampling_frequency: Sampling frequency
            - n_electrodes: Number of electrodes
            - cov: Whether to use coefficient of variation for fitness
    """
    with open(log_config_path, 'r') as f:
        log_data = json.load(f)
    
    # Load relevant data from log file
    movement_config = log_data['InputData']['Configuration']['MovementConfiguration']
    movement_type = movement_config['MovementType']
    effort_profile = movement_config['MovementProfileParameters']['EffortProfile']
    sampling_frequency = log_data['InputData']['Configuration']['RecordingConfiguration']['SamplingFrequency']
    n_electrodes = log_data['InputData']['Configuration']['RecordingConfiguration']['ElectrodeConfiguration']['NElectrodes']
    
    # Set use_coeff_var_fitness to True if movement is isometric
    cov = True
    if movement_type != 'Isometric':
        cov = False
    
    # Set timing parameters
    rest_duration = movement_config['MovementProfileParameters']['RestDuration']
    ramp_duration = movement_config['MovementProfileParameters']['RampDuration']
    start_time = rest_duration
    end_time = -rest_duration

    if effort_profile == 'Trapezoid':
        start_time = rest_duration + ramp_duration
        end_time = -rest_duration - ramp_duration
    
    return {
        'start_time': int(start_time),
        'end_time': int(end_time),
        'sampling_frequency': int(sampling_frequency),
        'low_pass_cutoff': 500,
        'n_electrodes': int(n_electrodes),
        'cov': cov
    }


def get_experimental_config(edf_path: Path) -> dict:
    """
    Extract configuration from experimental EDF file.
    
    Args:
        edf_path (Path): Path to the EDF file
        
    Returns:
        dict: Configuration dictionary containing:
            - start_time: Start time for analysis
            - end_time: End time for analysis
            - sampling_frequency: Sampling frequency
            - n_electrodes: Number of electrodes
            - cov: Whether to use coefficient of variation for fitness
    """
    sig = edfio.read_edf(edf_path)
    channels_df = pd.read_csv(edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_channels.tab", delimiter='\t')
    
    # Extract MVC from filename
    filename = edf_path.stem
    mvc_match = re.search(r'(\d+)percentmvc', filename)
    if not mvc_match:
        raise ValueError(f"Could not find MVC value in filename: {filename}")
    
    # Get the first non-None group (either from isometricXpercentmvc or _Xmvc)
    mvc = int(next(g for g in mvc_match.groups() if g is not None))
    
    # Find the requested path channel and get start/end times
    path_idx = channels_df.query('description.str.lower() == "requested path"').index[0]
    idx = np.where(np.diff(sig.signals[path_idx].data == mvc) == 1)[0]
    if len(idx) < 2:
        raise ValueError(f"Could not find MVC period in signal for {filename}")
    start_time = idx[0]
    end_time = idx[1]
    sampling_frequency = sig.signals[path_idx].sampling_frequency
    
    return {
        'start_time': int(start_time//sampling_frequency),
        'end_time': int(end_time//sampling_frequency),
        'sampling_frequency': int(sampling_frequency),
        'low_pass_cutoff': 4400 if sampling_frequency > 4400 else 500,
        'n_electrodes': len(channels_df[channels_df['type'].str.startswith('EMG')]),
        'cov': True
    }


def generate_algorithm_config(base_config_path: str, recording_config: dict) -> dict:
    """
    Generate algorithm configuration based on recording configuration.
    
    Args:
        base_config_path (str): Path to the base algorithm configuration file
        recording_config (dict): Configuration extracted from recording
        
    Returns:
        dict: Modified algorithm configuration
    """
    with open(base_config_path, 'r') as f:
        algo_config = json.load(f)
    
    # Update configuration with recording-specific parameters
    algo_config['Config']['start_time'] = recording_config['start_time']
    algo_config['Config']['end_time'] = recording_config['end_time']
    algo_config['Config']['sampling_frequency'] = recording_config['sampling_frequency']
    algo_config['Config']['extension_factor'] = int(1000 / recording_config['n_electrodes']) + 1
    algo_config['Config']['use_coeff_var_fitness'] = recording_config['cov']
    
    return algo_config


def process_recording(edf_path: Path, output_dir: Path, algorithm_config: str, 
                     container: str, data_type: str, data_config_path: Optional[Path] = None):
    """
    Process a single recording using EDF file and optional simulation log config.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        container (str): Path to the Singularity container
        data_type (str): Type of data ('simulated' or 'experimental')
        data_config_path (Optional[Path]): Path to the simulation log config file (for simulated data)
    """
    # Create output directory for this recording
    recording_output_dir = output_dir
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get recording configuration based on data type
    if data_type == 'simulated':
        recording_config = get_simulation_config(data_config_path)
    else:
        recording_config = get_experimental_config(edf_path)
    
    # Generate algorithm configuration
    algo_config = generate_algorithm_config(algorithm_config, recording_config)
    
    # Save modified config to output directory
    modified_config_path = recording_output_dir / 'algorithm_config.json'
    with open(modified_config_path, 'w') as f:
        json.dump(algo_config, f, indent=2)
    print(f"Saved modified algorithm config to {modified_config_path}")
    
    # Run decomposition
    try:
        decompose_scd(
            data=str(edf_path),
            output_dir=str(recording_output_dir),
            algorithm_config=str(modified_config_path),
            engine='singularity',
            container=container
        )
        print(f"Successfully decomposed recording")
        
        # Find the log file in the output directory
        log_files = list(recording_output_dir.glob('*_log.json'))
        output_log_path = log_files[0]
        print(f"Found log file: {output_log_path}")
        
        with open(output_log_path, 'r') as f:
            output_log = json.load(f)
        
        # Check runtime environment for GPU availability
        runtime_env = output_log.get('RuntimeEnvironment', {})
        gpu_list = runtime_env.get('Host', {}).get('GPU', [])
        
        # Update device in algorithm configuration
        if gpu_list and len(gpu_list) > 0:
            output_log['AlgorithmConfiguration']['Config']['device'] = 'cuda'
            print(f"GPU detected: {gpu_list[0]}, updating log with 'cuda' device")
        else:
            output_log['AlgorithmConfiguration']['Config']['device'] = 'cpu'
            print("No GPU detected, updating log with 'cpu' device")
        
        # Save updated log file
        with open(output_log_path, 'w') as f:
            json.dump(output_log, f, indent=2)
        print(f"Updated device information in {output_log_path}")
            
    except Exception as e:
        print(f"Error processing recording: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Decompose EMG recordings using SCD algorithm')
    parser.add_argument('--bids_root', default=BIDS_ROOT,
                      help='Root directory of the BIDS dataset')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                      help='Output directory for decomposition results')
    parser.add_argument('--algorithm_config', default=CONFIG,
                      help='Path to algorithm configuration file')
    parser.add_argument('--container', default=CONTAINER,
                      help='Path to Singularity container')
    parser.add_argument('--min_id', type=int, default=0,
                      help='Minimum ID to process (inclusive)')
    parser.add_argument('--max_id', type=int, default=None,
                      help='Maximum ID to process (inclusive). If None, process until the end.')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    
    # Create DataFrame of BIDS files
    try:
        df = pd.read_csv(bids_root / 'bids_dataframe.csv')
    except FileNotFoundError:
        df = create_bids_dataframe(bids_root)
        df.to_csv(bids_root / 'bids_dataframe.csv', index=False)

    print(f"\nFound {len(df)} recordings in BIDS structure")
    
    # Determine the range of IDs to process
    max_id = args.max_id if args.max_id is not None else len(df) - 1
    df_to_process = df.iloc[args.min_id:max_id + 1]
    
    print(f"\nProcessing recordings from ID {args.min_id} to {max_id} (total: {len(df_to_process)})")
    print("\nSample of files to process:")
    print(df_to_process.head())
    
    # Process each recording in the specified range
    for idx, row in df_to_process.iterrows():
        print(f"\nProcessing {Path(row['edf_path']).name} (ID: {idx})...")
        process_recording(
            edf_path=Path(row['edf_path']),
            output_dir=output_dir,
            algorithm_config=args.algorithm_config,
            container=args.container,
            data_type=row['data_type'],
            data_config_path=Path(row['log_config_path']) if row['data_type'] == 'simulated' else None
        )

if __name__ == '__main__':
    main()
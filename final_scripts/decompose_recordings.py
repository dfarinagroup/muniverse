#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
import edfio
from pathlib import Path
from muniverse.algorithms.decomposition import decompose_scd, decompose_cbss
from typing import Optional
import re


def extract_bids_components(filename: str) -> tuple[str, str, str]:
    """Extract BIDS components (subject, task) from filename."""
    # Define patterns for each component
    patterns = {
        'subject': r'sub-(?:sim)?(\d+)',
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


def generate_algorithm_config(base_config_path: str, recording_config: dict, algorithm: str) -> dict:
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
    if algorithm.lower() == 'scd':
        algo_config['Config']['extension_factor'] = int(1000 / recording_config['n_electrodes']) + 1
        algo_config['Config']['low_pass_cutoff'] = recording_config['low_pass_cutoff']
        algo_config['Config']['use_coeff_var_fitness'] = recording_config['cov']
    elif algorithm.lower() == 'cbss':
        algo_config['Config']['ext_fact'] = int(1000 / recording_config['n_electrodes']) + 1
        algo_config['Config']['bandpass'] = [20, recording_config['low_pass_cutoff']]
    
    return algo_config


def load_emg_data(edf_path: Path, data_type: str) -> np.ndarray:
    """
    Load and preprocess EMG data from EDF file.
    
    Args:
        edf_path: Path to the EDF file
        data_type: Type of data ('simulated' or 'experimental')
        
    Returns:
        np.ndarray: Preprocessed EMG data (channels x samples)
    """
    # Load EDF file
    raw = edfio.read_edf(edf_path)
    
    if data_type == 'experimental':
        # For experimental data, only keep EMG channels
        channels_df = pd.read_csv(edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_channels.tab", delimiter='\t')
        emg_channels = channels_df[channels_df['type'].str.startswith('EMG')].index
        data = np.stack([raw.signals[i].data for i in emg_channels])
    else:
        # For simulated data, use all channels
        data = np.stack([raw.signals[i].data for i in range(raw.num_signals)])
    
    return data


def process_scd_recording(edf_path: Path, output_dir: Path, algorithm_config: str, container: str, data_type: str) -> None:
    """
    Process a single recording using SCD algorithm.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        container (str): Path to the Singularity container
        data_type (str): Type of data ('simulated' or 'experimental')
    """
    # Create output directory for this recording
    recording_output_dir = output_dir
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save modified config to output directory
    modified_config_path = recording_output_dir / 'algorithm_config.json'
    with open(modified_config_path, 'w') as f:
        json.dump(algorithm_config, f, indent=2)
    print(f"Saved modified algorithm config to {modified_config_path}")
    
    # Load and preprocess data
    data = load_emg_data(edf_path, data_type)
    
    # Extract metadata for logging purposes
    metadata = {'filename': edf_path.name, 'format': 'edf'}
    
    # Run SCD decomposition
    decompose_scd(
        data=data,
        output_dir=str(recording_output_dir),
        algorithm_config=str(modified_config_path),
        engine='singularity',
        container=container,
        metadata=metadata  # Used by logger to track input data provenance
    )
    
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


def process_cbss_recording(edf_path: Path, output_dir: Path, algorithm_config: str, data_type: str) -> None:
    """
    Process a single recording using CBSS algorithm.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        data_type (str): Type of data ('simulated' or 'experimental')
    """
    # Create output directory for this recording
    recording_output_dir = output_dir
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save modified config to output directory
    modified_config_path = recording_output_dir / 'algorithm_config.json'
    with open(modified_config_path, 'w') as f:
        json.dump(algorithm_config, f, indent=2)
    print(f"Saved modified algorithm config to {modified_config_path}")

    # Load and preprocess data
    data = load_emg_data(edf_path, data_type)
    start_time = algorithm_config['Config']['start_time']
    end_time = algorithm_config['Config']['end_time']
    sf = algorithm_config['Config']['sampling_frequency']
    data = data[:, start_time*sf:end_time*sf]
    
    print(f"Data shape: {data.shape}")
    # Extract metadata for logging purposes
    metadata = {'filename': edf_path.name, 'format': 'edf'}
    
    # Run CBSS decomposition
    decompose_cbss(
        data=data,
        output_dir=str(recording_output_dir),
        algorithm_config=str(modified_config_path),
        metadata=metadata  # Used by logger to track input data provenance
    )


def process_recording(edf_path: Path, output_dir: Path, algorithm_config: str, 
                     container: Optional[str], data_type: str, data_config_path: Optional[Path] = None,
                     algorithm: str = 'scd'):
    """
    Process a single recording using EDF file and optional simulation log config.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        container (Optional[str]): Path to the Singularity container (only needed for SCD)
        data_type (str): Type of data ('simulated' or 'experimental')
        data_config_path (Optional[Path]): Path to the simulation log config file (for simulated data)
        algorithm (str): Algorithm to use ('scd' or 'cbss')
    """
    try:
        # Get recording configuration based on data type
        if data_type == 'simulated':
            recording_config = get_simulation_config(data_config_path)
        else:
            recording_config = get_experimental_config(edf_path)
        
        # Generate algorithm configuration
        algo_config = generate_algorithm_config(algorithm_config, recording_config, algorithm)
        
        # Route to appropriate processing function
        if algorithm.lower() == 'scd':
            if container is None:
                raise ValueError("Container path is required for SCD algorithm")
            process_scd_recording(edf_path, output_dir, algo_config, container, data_type)
        elif algorithm.lower() == 'cbss':
            process_cbss_recording(edf_path, output_dir, algo_config, data_type)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        print(f"Successfully decomposed recording using {algorithm.upper()}")
            
    except Exception as e:
        print(f"Error processing recording: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Decompose EMG recordings using SCD or CBSS algorithm')
    parser.add_argument('-d', '--dataset_name', help='Name of the dataset to process')
    parser.add_argument('-a', '--algorithm', choices=['scd', 'cbss'], help='Algorithm to use for decomposition')
    parser.add_argument('--container', help='Path to Singularity container (only needed for SCD)')
    parser.add_argument('--min_id', type=int, default=0,
                      help='Minimum ID to process (inclusive)')
    parser.add_argument('--max_id', type=int, default=None,
                      help='Maximum ID to process (inclusive). If None, process until the end.')
    
    args = parser.parse_args()
    
    # Set default paths based on algorithm choice
    DATASET_NAME = args.dataset_name
    BIDS_ROOT = f'/rds/general/user/pm1222/ephemeral/muniverse/datasets/bids/{DATASET_NAME}'
    OUTPUT_DIR = f'/rds/general/user/pm1222/ephemeral/muniverse/interim/{args.algorithm}_outputs/{DATASET_NAME}'

    SCD_CONFIG = f'/rds/general/user/pm1222/home/muniverse-demo/configs/scd.json'
    CBSS_CONFIG = f'/rds/general/user/pm1222/home/muniverse-demo/configs/cbss.json'    
    CONTAINER = args.container or f'/rds/general/user/pm1222/home/muniverse-demo/environment/muniverse_scd.sif'
    
    # Set default config and container based on algorithm choice
    if args.algorithm_config is None:
        args.algorithm_config = SCD_CONFIG if args.algorithm == 'scd' else CBSS_CONFIG
    if args.algorithm == 'scd' and args.container is None:
        args.container = CONTAINER
    
    # Convert paths to Path objects
    bids_root = Path(BIDS_ROOT)
    output_dir = Path(OUTPUT_DIR)
    
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
    print(f"Using {args.algorithm.upper()} algorithm")
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
            data_config_path=Path(row['log_config_path']) if row['data_type'] == 'simulated' else None,
            algorithm=args.algorithm
        )

if __name__ == '__main__':
    main()
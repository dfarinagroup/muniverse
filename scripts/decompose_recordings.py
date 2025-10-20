#!/usr/bin/env python3
"""
This script was used to decompose the EMG recordings for the MUniverse benchmark.

Usage:
    python decompose_recordings.py -d <dataset> -a <algorithm>

Example:
    python decompose_recordings.py -d Neuromotion-test -a scd

CLI Arguments:
    -d, --dataset: Name of the dataset to process
    -a, --algorithm: Algorithm to use for decomposition
    --min_id: Minimum recording index to process (inclusive). If not provided, only max_id is used.
    --max_id: Maximum recording index to process (inclusive). If min_id not provided, processes first N pending recordings. If both provided, uses indices from full inventory and skips already-processed.
    --inventory: Path to inventory CSV file. If not provided, uses default location.
"""

import json
import argparse
import time
import numpy as np
import pandas as pd
import edfio
from pathlib import Path
from typing import Optional, Dict
import re
import traceback

from muniverse.algorithms.decomposition import decompose_scd, decompose_cbss, decompose_upperbound
from muniverse.algorithms.upperbound import process_neuromotion_muaps, process_hybrid_tibialis_muaps
from muniverse.algorithms.core import spike_dict_to_long_df
from bookkeeping import update_inventory_status

def adjust_spike_times(spike_dict: Dict, time_offset_samples: int) -> Dict:
    """
    Adjust spike times by adding a time offset.
    
    This is necessary because decomposition algorithms operate on sliced data
    (e.g., data[:, start_time:end_time]), so spike indices are relative to the
    slice start. We need to convert them back to the original recording timeframe.
    """
    adjusted_spikes = {}
    for unit_id, spike_samples in spike_dict.items():
        # Add offset to each spike time
        adjusted_spikes[unit_id] = [int(s + time_offset_samples) for s in spike_samples]
    return adjusted_spikes


def get_simulation_config(edf_path: Path) -> Dict:
    """Extract relevant parameters from simulation log file.
    """
    with open(edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_simulation.json", 'r') as f:
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


def get_experimental_config(edf_path: Path) -> Dict:
    """Extract relevant parameters from experimental config file, i.e., channels.tab.
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


def generate_algorithm_config(base_config_path: str, recording_config: Dict, algorithm: str) -> Dict:
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
        algo_config['Config']['extension_factor'] = 10
        algo_config['Config']['low_pass_cutoff'] = recording_config['low_pass_cutoff']
        algo_config['Config']['use_coeff_var_fitness'] = recording_config['cov']
    elif algorithm.lower() == 'cbss' or algorithm.lower() == 'upperbound':
        algo_config['Config']['ext_fact'] = 10
        algo_config['Config']['bandpass'] = [20, recording_config['low_pass_cutoff']]
    
    return algo_config


def save_decomposition_results(
    output_dir: Path, 
    results: Dict, 
    metadata: Dict, 
    fsamp: Optional[float] = None,
    time_offset_samples: int = 0
) -> None:
    """
    Save decomposition results and metadata 
    
    Args:
        output_dir: Base directory to save results
        results: Dictionary of decomposition results containing sources, spikes, silhouette, etc.
        metadata: Dictionary with processing metadata
        fsamp: Sampling frequency, needed for converting spikes to time format
        time_offset_samples: Sample offset to add to spike times (to convert from sliced
                           data timeframe to original recording timeframe)
    """    
    # Each decomposition outputs a maximum of 5 files
    # 1. Save spikes as TSV in long format if fsamp is provided
    if fsamp is not None and 'spikes' in results and results['spikes']:
        # Adjust spike times to original recording timeframe
        adjusted_spikes = adjust_spike_times(results['spikes'], time_offset_samples)
        
        # Convert to long format DataFrame
        spikes_df = spike_dict_to_long_df(adjusted_spikes, fsamp=fsamp)
        spikes_path = output_dir / 'predicted_timestamps.tsv'
        spikes_df.to_csv(spikes_path, sep='\t', index=False)
    
    # 2. Save sources as compressed NPZ
    if 'sources' in results:
        sources_path = output_dir / 'predicted_sources.npz'
        np.savez_compressed(sources_path, sources=results['sources'])
    
    # 3. Save silhouette scores as compressed NPZ
    # TODO: _run_scd.py can be modified to output silhouette scores
    if 'silhouette' in results:
        sil_path = output_dir / 'silhouette.npz'
        np.savez_compressed(sil_path, silhouette=results['silhouette'])
    
    # 4. Save MU filters as compressed NPZ
    if 'mu_filters' in results:
        filters_path = output_dir / 'mu_filters.npz'
        np.savez_compressed(filters_path, mu_filters=results['mu_filters'])
    
    # 5. Save metadata as JSON (keeping this for compatibility)
    metadata_path = output_dir / "process_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir


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
    Process a single recording using SCD algorithm and save the results.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        container (str): Path to the Singularity container
        data_type (str): Type of data ('simulated' or 'experimental')
    """    
    # Create output directory for this recording
    recording_output_dir = output_dir / edf_path.stem[:-4]
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    data = load_emg_data(edf_path, data_type)

    # Extract parameters from config to compute time offset
    start_time = algorithm_config['Config']['start_time']
    sf = algorithm_config['Config']['sampling_frequency']
    time_offset_samples = int(start_time * sf)
    
    # Extract metadata for logging purposes
    metadata = {'filename': edf_path.name, 'format': 'edf'}
    
    # Run SCD decomposition
    results, metadata = decompose_scd(
        data=data,
        algorithm_config=algorithm_config,
        engine='singularity',
        container=container,
        metadata=metadata  # Used by logger to track input data provenance
    )
    
    # Check runtime environment for GPU availability
    runtime_env = metadata.get('RuntimeEnvironment', {})
    gpu_list = runtime_env.get('Host', {}).get('GPU', [])
    
    # Update device in algorithm configuration
    if gpu_list and len(gpu_list) > 0:
        metadata['AlgorithmConfiguration']['Config']['device'] = 'cuda'
        print(f"GPU detected: {gpu_list[0]}, updating log with 'cuda' device")
    else:
        metadata['AlgorithmConfiguration']['Config']['device'] = 'cpu'
        print("No GPU detected, updating log with 'cpu' device")
    
    save_decomposition_results(
        recording_output_dir, 
        results, 
        metadata, 
        fsamp=sf,
        time_offset_samples=time_offset_samples
    )


def process_cbss_recording(edf_path: Path, output_dir: Path, algorithm_config: str, data_type: str) -> None:
    """
    Process a single recording using CBSS algorithm and save the results.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        data_type (str): Type of data ('simulated' or 'experimental')
    """
    # Create output directory for this recording
    recording_output_dir = output_dir / edf_path.stem[:-4]
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EMG data
    data = load_emg_data(edf_path, data_type)
    
    # Extract parameters from config to compute time offset
    start_time = algorithm_config['Config']['start_time']
    sf = algorithm_config['Config']['sampling_frequency']
    time_offset_samples = int(start_time * sf)
    
    # Extract metadata for logging purposes
    metadata = {'filename': edf_path.name, 'format': 'edf'}
    
    # Run CBSS decomposition (function slices data internally based on config)
    results, metadata = decompose_cbss(
        data=data,
        algorithm_config=algorithm_config,
        metadata=metadata
    )
    
    # Save results with time offset adjustment
    save_decomposition_results(
        recording_output_dir, 
        results, 
        metadata, 
        fsamp=sf,
        time_offset_samples=time_offset_samples
    )


def process_upperbound_recording(edf_path: Path, output_dir: Path, algorithm_config: str, data_type: str, dataset: str) -> None:
    """
    Process a single recording using Upperbound algorithm and save the results.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (Optional[str]): Path to the algorithm configuration file
        data_type (str): Type of data ('simulated' or 'experimental')
        dataset (str): Name of the dataset to process
    """
    # Set default MUAP cache directory
    MUAP_CACHE_DIR = '/rds/general/user/pm1222/home/data/muapcache/'
    muap_cache_dir = Path(MUAP_CACHE_DIR)
    
    # Create output directory for this recording
    recording_output_dir = output_dir / edf_path.stem[:-4]
    recording_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EMG data
    data = load_emg_data(edf_path, data_type)
    
    # Extract parameters from config to compute time offset
    start_time = algorithm_config['Config']['start_time']
    sf = algorithm_config['Config']['sampling_frequency']
    time_offset_samples = int(start_time * sf)
    
    simulation_file = edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_simulation.json"
    with open(simulation_file, 'r') as f:
        simulation_config = json.load(f)
    
    subject_id = simulation_config['InputData']['Configuration']['SubjectConfiguration']['SubjectSeed']
    movement_config = simulation_config['InputData']['Configuration']['MovementConfiguration']
    movement_dof = movement_config['MovementDOF']
    muscle = movement_config['TargetMuscle']
    
    if dataset == 'Neuromotion-test':
        muap_cache_file = muap_cache_dir / f'subject_{subject_id}_{muscle}_{movement_dof}_muaps.npy'
        muaps = np.load(muap_cache_file)
        processed_muaps = process_neuromotion_muaps(muaps, simulation_config)
    elif dataset == 'Hybrid-Tibialis':
        muap_cache_file = muap_cache_dir / f'hybrid_TA_muaps.npz'
        muaps = np.load(muap_cache_file)
        processed_muaps = process_hybrid_tibialis_muaps(muaps, simulation_config)
        
    # Extract metadata for logging purposes
    metadata = {'filename': edf_path.name, 'format': 'edf', 'data_type': data_type}
    
    # Run upperbound decomposition (function slices data internally based on config)
    results, metadata = decompose_upperbound(
        data=data,
        muaps=processed_muaps,
        algorithm_config=algorithm_config,
        metadata=metadata
    )
    
    # Save results with time offset adjustment
    save_decomposition_results(
        recording_output_dir, 
        results, 
        metadata, 
        fsamp=sf,
        time_offset_samples=time_offset_samples
    )


def process_recording(edf_path: Path, output_dir: Path, algorithm_config: str, 
                     container: Optional[str], data_type: str,
                     algorithm: str = 'scd', dataset: str = 'Neuromotion-test'):
    """
    Process a single recording using EDF file and optional simulation log config.
    
    Args:
        edf_path (Path): Path to the EDF file
        output_dir (Path): Base output directory for decomposition results
        algorithm_config (str): Path to the algorithm configuration file
        container (Optional[str]): Path to the Singularity container (only needed for SCD)
        data_type (str): Type of data ('simulated' or 'experimental')
        algorithm (str): Algorithm to use ('scd', 'cbss', or 'upperbound')
        dataset (str): Name of the dataset to process
    """
    try:
        # Get recording configuration based on data type
        if data_type == 'simulated':
            recording_config = get_simulation_config(edf_path)
        else:
            recording_config = get_experimental_config(edf_path)
        
        algo_config = generate_algorithm_config(algorithm_config, recording_config, algorithm)

        # Route to appropriate processing function based on algorithm
        if algorithm.lower() == 'scd':
            if container is None:
                raise ValueError("Container path is required for SCD algorithm")
            process_scd_recording(edf_path, output_dir, algo_config, container, data_type)
        elif algorithm.lower() == 'cbss':
            process_cbss_recording(edf_path, output_dir, algo_config, data_type)
        elif algorithm.lower() == 'upperbound':
            process_upperbound_recording(edf_path, output_dir, algo_config, data_type, dataset)
            
        print(f"Successfully decomposed recording using {algorithm.upper()}")
            
    except Exception as e:
        print(f"Error processing recording: {str(e)}")
        print(traceback.format_exc())
        raise  # Re-raise so outer try-except can handle status tracking


def main():
    parser = argparse.ArgumentParser(description='Decompose EMG recordings using SCD, CBSS, or Upperbound algorithm')
    parser.add_argument('-d', '--dataset', required=True, help='Name of the dataset to process')
    parser.add_argument('-a', '--algorithm', required=True, choices=['scd', 'cbss', 'upperbound'], 
                       help='Algorithm to use for decomposition')
    parser.add_argument('--min_id', type=int, default=None,
                       help='Minimum recording index to process (inclusive). If not provided, only max_id is used.')
    parser.add_argument('--max_id', type=int, default=None,
                       help='Maximum recording index to process (inclusive). If min_id not provided, processes first N pending recordings. If both provided, uses indices from full inventory and skips already-processed.')
    parser.add_argument('--inventory', type=Path, default=None,
                       help='Path to inventory CSV file. If not provided, uses default location.')
    
    args = parser.parse_args()
    
    # Set default paths
    INTERIM_DIR = '/rds/general/user/pm1222/ephemeral/muniverse/interim'
    CONFIG_DIR = '/rds/general/user/pm1222/home/muniverse/configs'
    CONTAINER = '/rds/general/user/pm1222/home/muniverse/environment/muniverse_scd.sif'
    
    # Determine output directory and inventory path
    output_base_dir = Path(INTERIM_DIR) / f"{args.algorithm}" / args.dataset
    
    if args.inventory is None:
        inventory_path = output_base_dir / f'inventory_{args.algorithm}.csv'
    else:
        inventory_path = args.inventory
    
    # Check if inventory exists
    if not inventory_path.exists():
        print(f"Error: Inventory not found at {inventory_path}")
        print(f"Create it first using: python bookkeeping.py -d {args.dataset} -a {args.algorithm} --create")
        return
    
    # Load inventory with retry logic for concurrent access
    print(f"Loading inventory from: {inventory_path}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            inventory = pd.read_csv(inventory_path)
            break
        except (PermissionError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            if attempt < max_retries - 1:
                print(f"Retrying inventory read (attempt {attempt + 1}/{max_retries})...")
                time.sleep(0.5 * (1.5 ** attempt))
            else:
                raise RuntimeError(f"Failed to read inventory after {max_retries} attempts: {e}")
    
    # Two modes of operation:
    # 1. If only max_id provided: Take first N pending recordings (for processing all pending in chunks)
    # 2. If both min_id and max_id provided: Use indices from full inventory, skip non-pending (idempotent)
    
    if args.min_id is None:
        # Mode 1: Process first N pending recordings
        pending_recordings = inventory[inventory['status'].isin(['pending', 'failed'])]
        
        if len(pending_recordings) == 0:
            print("No pending recordings to process!")
            return
        
        max_id = args.max_id if args.max_id is not None else len(pending_recordings) - 1
        recordings_to_process = pending_recordings.iloc[0:max_id + 1]
        
        print(f"\nFound {len(pending_recordings)} pending recordings")
        print(f"Processing first {len(recordings_to_process)} pending recordings")
        print(f"Using {args.algorithm.upper()} algorithm\n")
    else:
        # Mode 2: Use indices from full inventory, filter to pending/failed
        # This makes parallel jobs idempotent - they can safely overlap
        max_id = args.max_id if args.max_id is not None else len(inventory) - 1
        selected_recordings = inventory.iloc[args.min_id:max_id + 1]
        
        # Filter to only pending/failed (skip already processed)
        recordings_to_process = selected_recordings[selected_recordings['status'].isin(['pending', 'failed'])]
        
        print(f"\nTotal recordings in inventory: {len(inventory)}")
        print(f"Selected range [{args.min_id}:{max_id}]: {len(selected_recordings)} recordings")
        print(f"Of which {len(recordings_to_process)} are pending/failed (rest already processed)")
        print(f"Using {args.algorithm.upper()} algorithm\n")
        
        if len(recordings_to_process) == 0:
            print("All recordings in specified range already processed!")
            return

    # Mark all recordings as processing
    for _, row in recordings_to_process.iterrows():
        recording_id = row['recording_id']
        # Check current status before marking as processing
        current_status = inventory[inventory['recording_id'] == recording_id]['status'].values[0]
        if current_status in ['pending', 'failed']:
            update_inventory_status(inventory_path, recording_id=recording_id, status='processing')
    
    # Set algorithm config path
    algorithm_config = str(Path(CONFIG_DIR) / f"{args.algorithm}.json")
    
    # Process each recording
    for _, row in recordings_to_process.iterrows():
        recording_id = row['recording_id']
        edf_path = Path(row['edf_path'])        
        print(f"Processing recording {recording_id}: {edf_path.name}")
        
        try:
            # Process the recording
            process_recording(
                edf_path=edf_path,
                output_dir=output_base_dir,
                algorithm_config=algorithm_config,
                container=CONTAINER if args.algorithm == 'scd' else None,
                data_type=row['data_type'],
                algorithm=args.algorithm,
                dataset=args.dataset
            )
            
            # Mark as completed
            update_inventory_status(inventory_path, recording_id=recording_id, output_path=output_base_dir / edf_path.stem[:-4], status='completed')
            print(f"Successfully completed recording {recording_id}")
            
        except Exception as e:
            error_msg = f"{str(e)[:200]}"
            print(f"Failed to process recording {recording_id}: {error_msg}")
            
            # Save detailed error log
            error_log_path = output_base_dir / edf_path.stem[:-4] / 'error.log'
            with open(error_log_path, 'w') as f:
                f.write(f"Error processing {edf_path.name}:\n")
                f.write(traceback.format_exc())
            
            # Mark as failed in inventory
            update_inventory_status(
                inventory_path, 
                recording_id=recording_id, 
                status='failed',
                error_message=error_msg
            )

if __name__ == '__main__':
    main()
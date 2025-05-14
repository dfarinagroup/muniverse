#!/usr/bin/env python3
# run_neuromotion_curated.py - Version that uses curated MUAPs with MotoneuronPool

import argparse
import os
import torch
import time
import json
import numpy as np
import random
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt
from tqdm import tqdm

import sys
sys.path.append('.')

from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_params import mn_default_settings

def set_seed(seed):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Try to make PyTorch operations deterministic
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:  # Fallback for YAML if needed
            import yaml
            config = yaml.safe_load(f)
    return edict(config)

def create_trapezoid_effort(fs, movement_duration, effort_level, rest_duration, ramp_duration, hold_duration):
    """Create a trapezoidal effort profile."""
    # One contraction consists of rest - ramp up - hold - ramp down - rest
    rest_samples = round(fs * rest_duration)
    ramp_samples = round(fs * ramp_duration)
    hold_samples = round(fs * hold_duration)
    
    muscle_force = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, effort_level, ramp_samples),
        np.ones(hold_samples) * effort_level,
        np.linspace(effort_level, 0, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    # Add an extra second at the end (zero padding)
    extra_samples = round(fs * 1.0)  # 1 second
    muscle_force = np.concatenate([muscle_force, np.zeros(extra_samples)])

    # Ensure the profile doesn't exceed the specified duration
    expected_samples = round(fs * movement_duration)
    if len(muscle_force) > expected_samples:
        muscle_force = muscle_force[:expected_samples]
    elif len(muscle_force) < expected_samples:
        # Pad with zeros if shorter than expected
        muscle_force = np.pad(muscle_force, (0, expected_samples - len(muscle_force)), 'constant')
    
    return muscle_force

def create_triangular_effort(fs, movement_duration, effort_level, rest_duration, ramp_duration, n_reps=1):
    """Create a triangular effort profile with specified parameters."""
    # One contraction consists of rest - ramp up - ramp down - rest
    rest_samples = round(fs * rest_duration)
    ramp_samples = round(fs * ramp_duration)
    one_contraction = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, effort_level, ramp_samples),
        np.linspace(effort_level, 0, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    # Repeat the contraction pattern n_reps times
    muscle_force = np.tile(one_contraction, n_reps)
    
    # Add an extra second at the end (zero padding)
    extra_samples = round(fs * 1.0)  # 1 second
    muscle_force = np.concatenate([muscle_force, np.zeros(extra_samples)])

    # Ensure the profile doesn't exceed the specified duration
    expected_samples = round(fs * movement_duration)
    if len(muscle_force) > expected_samples:
        muscle_force = muscle_force[:expected_samples]
    elif len(muscle_force) < expected_samples:
        # Pad with zeros if shorter than expected
        muscle_force = np.pad(muscle_force, (0, expected_samples - len(muscle_force)), 'constant')
    
    return muscle_force

def exponential_sample_motor_units_by_index(muaps, num_to_select, seed, exp_factor=5.0):
    """
    Sample motor units with an exponential bias toward lower indices.
    Assumes MUAPs are already sorted from smallest to largest.
    
    Args:
        muaps (ndarray): MUAPs with shape (n_motor_units, electrodes, samples)
        num_to_select (int): Number of motor units to select
        seed (int): Random seed for reproducible selection
        exp_factor (float): Exponential factor - higher values increase bias toward lower indices
    
    Returns:
        tuple: (selected_muaps, selected_indices)
    """
    # Create a dedicated RNG for this function to ensure reproducibility
    rng = np.random.RandomState(seed)
    
    num_mus = len(muaps)
    
    if num_to_select >= num_mus:
        print(f"Warning: Requested {num_to_select} MUs but only {num_mus} available. Using all available MUs.")
        return muaps, np.arange(num_mus)
    
    # Generate exponential weights - higher probability for lower indices
    weights = np.exp(-exp_factor * np.arange(num_mus) / num_mus)
    weights = weights / np.sum(weights)
    
    # Sample the desired number of motor units
    selected_indices = rng.choice(num_mus, size=num_to_select, replace=False, p=weights)
    
    # Sort the indices to maintain the order
    selected_indices = np.sort(selected_indices)
    
    selected_muaps = muaps[selected_indices]
    
    # Calculate quartiles of the selected indices as a percentage of total MUs
    index_percentages = selected_indices * 100 / num_mus
    print(f"Selected {num_to_select} motor units with index range: [{selected_indices.min()}-{selected_indices.max()}] of {num_mus}")
    print(f"Index percentiles: 25%={np.percentile(index_percentages, 25):.1f}%, "
          f"50%={np.percentile(index_percentages, 50):.1f}%, "
          f"75%={np.percentile(index_percentages, 75):.1f}%")
    
    return selected_muaps, selected_indices

def generate_spike_trains(mn_pool, effort_profile, fs):
    """Generate spike trains based on an effort profile using MotoneuronPool."""
    # Initialize the motoneuron pool
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()
    
    # Generate spike trains
    ext_new, spikes, fr, ipis = mn_pool.generate_spike_trains(effort_profile, fit=False)
    
    return ext_new, spikes, fr, ipis

def generate_emg_signal(muaps, spikes, time_samples, noise_level_db=None, noise_seed=None):
    """
    Generate EMG signal by convolving MUAPs with spike trains.
    
    Args:
        muaps (numpy.ndarray): MUAPs with shape (num_mus, electrodes, samples).
        spikes (list): List of spike trains for each motor unit.
        time_samples (int): Number of time samples in the effort profile.
        noise_level_db (float, optional): Noise level in dB. If None, no noise is added.
        noise_seed (int, optional): Random seed for noise generation.
    
    Returns:
        numpy.ndarray: EMG signal with shape (samples, electrodes).
    """
    start_time = time.time()
    num_mus = len(spikes)
    win = muaps.shape[2]  # Time samples in each MUAP
    offset = win // 2
    
    # Determine number of active motor units
    units_active = 0
    for sp in spikes:
        if len(sp) > 0:
            units_active += 1
    
    # Initialize EMG signal
    num_electrodes = muaps.shape[1]
    emg = np.zeros((time_samples, num_electrodes))
    
    # Add contribution from each motor unit
    for unit in tqdm(range(units_active), desc="Convolving MUAPs with spikes"):
        unit_firings = spikes[unit]
        
        if len(unit_firings) == 0:
            continue
        
        for firing in unit_firings:
            # Get the MUAP for this unit
            muap = muaps[unit]
            
            # Determine time window overlap
            init_emg = max(0, firing - offset)
            end_emg = min(firing + offset, time_samples)
            
            init_muap = init_emg - (firing - offset)  # Start index in MUAP window
            end_muap = init_muap + (end_emg - init_emg)  # End index in MUAP window
            
            # Add contribution to EMG
            emg[init_emg:end_emg, :] += muap[:, init_muap:end_muap].T
    
    # Add noise if specified
    if noise_level_db is not None:
        # Only this part should depend on the noise seed rather than the subject seed
        # to allow different noise realizations with the same underlying signal
        original_state = np.random.get_state()
        if noise_seed is not None:
            np.random.seed(noise_seed)
        
        std_emg = emg.std()
        std_noise = std_emg * 10 ** (-noise_level_db / 20)
        noise = np.random.normal(0, std_noise, emg.shape)
        emg = emg + noise
        
        # Restore random state
        if noise_seed is not None:
            np.random.set_state(original_state)
    
    print(f"EMG generation completed in {time.time() - start_time:.2f} seconds")
    
    return emg

def load_muaps(muaps_file):
    """
    Load MUAPs from file.
    
    Args:
        muaps_file (str): Path to the MUAPs file
        
    Returns:
        numpy.ndarray: MUAPs with shape (n_motor_units, electrodes, samples)
    """
    try:
        print(f"Loading MUAPs from {muaps_file}")
        data = np.load(muaps_file)
        
        # Check available arrays in the file
        for key in data.keys():
            print(f"Found key in file: {key}")
        
        # Load MUAPs - expected key is 'muaps'
        if 'muaps' in data:
            muaps = data['muaps']
        else:
            # Try alternative keys if 'muaps' not found
            for alt_key in ['muap', 'MUAP', 'MUAPs']:
                if alt_key in data:
                    muaps = data[alt_key]
                    break
            else:
                raise KeyError("Could not find MUAPs array in the file")
        
        print(f"Loaded MUAPs with shape: {muaps.shape}")
        
        return muaps
    except Exception as e:
        print(f"Error loading MUAPs: {e}")
        raise

def create_effort_profile(fs, movement_duration, profile_params):
    """Create an effort profile based on the movement parameters."""
    effort_level = profile_params.EffortLevel / 100.0  # Convert percentage to decimal
    
    if hasattr(profile_params, 'EffortProfile'):
        if profile_params.EffortProfile == "Trapezoid":
            return create_trapezoid_effort(
                fs, 
                movement_duration, 
                effort_level, 
                profile_params.RestDuration, 
                profile_params.RampDuration, 
                profile_params.HoldDuration
            )
        elif profile_params.EffortProfile == "Triangular":
            n_reps = getattr(profile_params, 'NRepetitions', 1)
            return create_triangular_effort(
                fs,
                movement_duration,
                effort_level,
                profile_params.RestDuration,
                profile_params.RampDuration,
                n_reps
            )
    
    # Default case - use trapezoid if not specified
    rest_duration = getattr(profile_params, 'RestDuration', 1.0)
    ramp_duration = getattr(profile_params, 'RampDuration', 2.0)
    hold_duration = getattr(profile_params, 'HoldDuration', 3.0)
    
    return create_trapezoid_effort(
        fs, 
        movement_duration, 
        effort_level, 
        rest_duration, 
        ramp_duration, 
        hold_duration
    )

def save_outputs(output_dir, emg, spikes, ext, cfg, metadata):
    """Save all outputs to the specified directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get subject ID for filename prefix
    subject_id = metadata["simulation_info"].get("subject_id", "")
    subject_prefix = f"{subject_id}_" if subject_id else ""
    muscle = metadata["simulation_info"].get("target_muscle", "")
    
    # Prepare output paths with subject ID prefix
    paths = {
        'emg': os.path.join(output_dir, f'{subject_prefix}{muscle}_emg.npz'),
        'spikes': os.path.join(output_dir, f'{subject_prefix}{muscle}_spikes.npz'),
        'effort_profile': os.path.join(output_dir, f'{subject_prefix}{muscle}_effort_profile.npz'),
        'config': os.path.join(output_dir, f'{subject_prefix}{muscle}_config_used.json'),
        'metadata': os.path.join(output_dir, f'{subject_prefix}{muscle}_metadata.json')
    }
    
    # Save each array as a separate compressed file
    np.savez_compressed(paths['emg'], emg=emg)
    np.savez_compressed(paths['spikes'], spikes=np.array(spikes, dtype=object))
    np.savez_compressed(paths['effort_profile'], effort_profile=ext)
    
    # Save configuration and metadata as JSON
    with open(paths['config'], 'w') as f:
        json.dump(cfg, f, indent=2)
    
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary of saved files
    print(f"Data saved to:")
    for key, path in paths.items():
        print(f"- {key}: {path}")
    
    return paths

def get_deterministic_mu_count(seed, min_mus=300, max_mus=350):
    """
    Get a deterministic number of motor units based on seed.
    
    Args:
        seed (int): Random seed
        min_mus (int): Minimum number of motor units
        max_mus (int): Maximum number of motor units
        
    Returns:
        int: Number of motor units
    """
    # Create dedicated RNG just for this operation
    rng = np.random.RandomState(seed)
    return rng.randint(min_mus, max_mus + 1)

def main():
    parser = argparse.ArgumentParser(description='Generate EMG signals from curated MUAPs')
    parser.add_argument('config_path', type=str, help='Path to input configuration JSON file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--muaps_file', type=str, required=True, 
                        help='Path to MUAPs file (npz format)')
    parser.add_argument('--pytorch-device', type=str, choices=['cpu', 'cuda'], default='cpu', 
                        help='PyTorch device to use')
    parser.add_argument('--exp-factor', type=float, default=5.0, 
                        help='Exponential sampling factor (higher = more bias toward low indices)')
    args = parser.parse_args()

    # Set device for PyTorch
    device = args.pytorch_device
    print(f"Using PyTorch device: {device}")
    
    # Load configuration
    cfg = load_config(args.config_path)
    
    # Subject configuration
    subject_cfg = cfg.SubjectConfiguration
    subject_seed = subject_cfg.SubjectSeed
    subject_id = subject_cfg.get('SubjectID', f"subject_{subject_seed}")
    
    # Set all random seeds for reproducibility
    set_seed(subject_seed)
    print(f"All random seeds set to {subject_seed} for reproducibility")
    
    # Movement configuration
    movement_cfg = cfg.MovementConfiguration
    ms_label = movement_cfg.TargetMuscle
    if ms_label != "Tibialis Anterior":
        print(f"Warning: This script is optimized for Tibialis Anterior, but '{ms_label}' was specified.")
    
    movement_duration = movement_cfg.MovementProfileParameters.MovementDuration
    
    # Recording configuration
    recording_cfg = cfg.RecordingConfiguration
    fs = recording_cfg.SamplingFrequency
    electrode_cfg = recording_cfg.ElectrodeConfiguration
    noise_seed = recording_cfg.NoiseSeed
    noise_level_db = recording_cfg.NoiseLeveldb
    
    # Load MUAPs from file
    all_muaps = load_muaps(args.muaps_file)
    
    # Get deterministic number of motor units for this subject
    num_mus = subject_cfg.get('MuscleMotorUnitCounts', 340)[0]
    
    # Select motor units using exponential sampling based on index
    print(f"Selecting {num_mus} motor units with exponential sampling (factor={args.exp_factor})...")
    selected_muaps, selected_indices = exponential_sample_motor_units_by_index(
        all_muaps, 
        num_mus, 
        subject_seed,  # Use subject seed directly for reproducibility
        exp_factor=args.exp_factor
    )
    
    # Determine electrode grid dimensions
    if hasattr(electrode_cfg, 'GridShape'):
        grid_rows, grid_cols = electrode_cfg.GridShape
        print(f"Using grid shape from config: {grid_rows}x{grid_cols}")
    else:
        # Try to determine from other parameters
        total_electrodes = getattr(electrode_cfg, 'EMGChannelCount', selected_muaps.shape[1])
        
        if hasattr(electrode_cfg, 'NGrids') and hasattr(electrode_cfg, 'GridShape'):
            # If we have multiple grids with the same shape
            grid_rows, grid_cols_per_grid = electrode_cfg.GridShape
            grid_cols = grid_cols_per_grid * electrode_cfg.NGrids
        else:
            # Default for Tibialis Anterior based on the config example
            grid_rows = 13
            grid_cols = 5 * getattr(electrode_cfg, 'NGrids', 4)  # Default 4 grids
            
            # Verify this matches the total electrode count
            if grid_rows * grid_cols != total_electrodes:
                # Fallback to deriving grid shape from electrode count
                grid_rows = int(np.sqrt(total_electrodes))
                grid_cols = total_electrodes // grid_rows
                if grid_rows * grid_cols != total_electrodes:
                    grid_cols += 1
        
        print(f"Derived grid shape: {grid_rows}x{grid_cols}")
    
    # Create effort profile based on the configuration
    profile_params = movement_cfg.MovementProfileParameters
    effort_profile = create_effort_profile(fs, movement_duration, profile_params)
    
    effort_type = getattr(profile_params, 'EffortProfile', 'Trapezoid')
    effort_level = profile_params.EffortLevel
    print(f"Creating {effort_type} effort profile with level: {effort_level:.1f}%")
    
    # Initialize MotoneuronPool with dummy muscle 'ECRB'
    print(f"Initializing MotoneuronPool with {num_mus} motor units (dummy muscle: ECRB)...")
    mn_pool = MotoneuronPool(num_mus, 'ECRB', **mn_default_settings)
    
    # Generate spike trains using MotoneuronPool
    print("Generating spike trains using MotoneuronPool...")
    _, spikes, fr, ipis = generate_spike_trains(mn_pool, effort_profile, fs)
    
    # Generate EMG signal
    print("Generating EMG signal from spikes and MUAPs...")
    emg = generate_emg_signal(
        selected_muaps, 
        spikes, 
        len(effort_profile), 
        noise_level_db, 
        noise_seed
    )
    
    print(f"Generated EMG shape: {emg.shape}")
    
    # Create metadata
    metadata = {
        "simulation_info": {
            "num_motor_units": num_mus,
            "target_muscle": ms_label,
            "fs": fs,
            "electrode_array": {
                "rows": grid_rows,
                "columns": grid_cols,
                "total_electrodes": grid_rows * grid_cols,
            },
            "noise_level_db": noise_level_db,
            "noise_seed": noise_seed,
            "movement_duration": movement_duration,
            "movement_dof": movement_cfg.MovementDOF,
            "movement_type": "Isometric",  # Always isometric for this script
            "effort_profile": effort_type,
            "effort_level": profile_params.EffortLevel,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subject_id": subject_id,
            "subject_seed": subject_seed,
            "muap_source": args.muaps_file,
            "exponential_sampling_factor": args.exp_factor,
            "selected_indices": selected_indices.tolist(),  # Just show the first 10 indices
            "index_distribution": {
                "min": int(selected_indices.min()),
                "max": int(selected_indices.max()),
                "quartiles": [
                    float(np.percentile(selected_indices, 25)),
                    float(np.percentile(selected_indices, 50)),
                    float(np.percentile(selected_indices, 75))
                ]
            }
        }
    }
    
    # Save all outputs
    save_outputs(
        args.output_dir, 
        emg, 
        spikes, 
        effort_profile, 
        cfg, 
        metadata
    )
    
    print("EMG generation complete.")

if __name__ == '__main__':
    main()
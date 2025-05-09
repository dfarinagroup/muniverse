#!/usr/bin/env python3
# run_neuromotion_curated.py - Version that uses pre-sorted MUAPs and direct spike generation

import argparse
import os
import torch
import time
import json
import numpy as np
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt
from tqdm import tqdm

import sys
sys.path.append('.')

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

def exponential_sample_motor_units(muaps, thresholds, num_to_select, rng, exp_factor=5.0):
    """
    Sample motor units with an exponential bias toward lower recruitment thresholds.
    
    Args:
        muaps (ndarray): Sorted MUAPs with shape (n_motor_units, electrodes, samples)
        thresholds (ndarray): Sorted thresholds with shape (n_motor_units,)
        num_to_select (int): Number of motor units to select
        rng (numpy.random.RandomState): Random number generator
        exp_factor (float): Exponential factor - higher values increase bias toward low thresholds
    
    Returns:
        tuple: (selected_muaps, selected_thresholds, selected_indices)
    """
    num_mus = len(muaps)
    
    if num_to_select >= num_mus:
        print(f"Warning: Requested {num_to_select} MUs but only {num_mus} available. Using all available MUs.")
        return muaps, thresholds, np.arange(num_mus)
    
    # Generate exponential weights - higher probability for lower indices
    weights = np.exp(-exp_factor * np.arange(num_mus) / num_mus)
    weights = weights / np.sum(weights)
    
    # Sample the desired number of motor units
    selected_indices = rng.choice(num_mus, size=num_to_select, replace=False, p=weights)
    
    # Sort the indices to maintain the order by threshold
    selected_indices = np.sort(selected_indices)
    
    selected_muaps = muaps[selected_indices]
    selected_thresholds = thresholds[selected_indices]
    
    # Display distribution of selected thresholds
    if selected_thresholds is not None:
        print(f"Selected {num_to_select} motor units with threshold range: [{selected_thresholds.min():.1f}%, {selected_thresholds.max():.1f}%]")
        print(f"Threshold quartiles: 25%={np.percentile(selected_thresholds, 25):.1f}%, "
              f"50%={np.percentile(selected_thresholds, 50):.1f}%, "
              f"75%={np.percentile(selected_thresholds, 75):.1f}%")
    
    return selected_muaps, selected_thresholds, selected_indices

def generate_spike_trains_direct(recruitment_thresholds, excitation, fs, settings=None, rng=None):
    """
    Generate spike trains directly using recruitment thresholds and firing rate properties.
    
    Args:
        recruitment_thresholds (ndarray): Recruitment thresholds for each motor unit (% MVC)
        excitation (ndarray): Excitation profile (% MVC)
        fs (float): Sampling frequency in Hz
        settings (dict, optional): Dictionary of settings for spike generation
        rng (numpy.random.RandomState, optional): Random number generator
    
    Returns:
        tuple: (modified_excitation, spikes, firing_rates, inter_pulse_intervals)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Default settings if none provided (taken from mn_default_settings)
    if settings is None:
        settings = {
            'rr': 50,             # Range of recruitment thresholds
            'rm': 0.75,           # Recruitment threshold of the first MU
            'rp': 100,            # Range of twitch tensions
            'pfr1': 40,           # Peak firing rate of the first MU
            'pfrd': 10,           # Difference between peak firing rates of first and last MUs
            'mfr1': 10,           # Minimum firing rate of the first MU
            'mfrd': 5,            # Difference between minimum firing rates of first and last MUs
            'gain': 30,           # Firing rate gain (Hz per % MVC)
            'c_ipi': 0.1,         # Coefficient of variation of interpulse intervals
        }
    
    # Number of motor units and time samples
    N = len(recruitment_thresholds)
    time_samples = len(excitation)
    
    # Reshape recruitment thresholds to column vector if needed
    rte = recruitment_thresholds.reshape(N, 1)
    
    # Initialize firing rate parameters for each motor unit
    # Minimum firing rate: decreases linearly with recruitment threshold
    min_fr_range = settings['mfrd']  # Difference between first and last MU
    minfr = np.linspace(settings['mfr1'], settings['mfr1'] - min_fr_range, N).reshape(N, 1)
    
    # Maximum firing rate: decreases with recruitment threshold
    max_fr_first = settings['pfr1']  # Peak firing rate of first MU
    max_fr_range = settings['pfrd']  # Difference between first and last MU
    maxfr = max_fr_first - max_fr_range * rte / np.max(rte)
    
    # Slope (gain) of firing rate vs. excitation
    slope_fr = np.ones((N, 1)) * settings['gain']
    
    # Calculate firing rates for each motor unit at each time point
    fr = np.zeros((N, time_samples))
    for t in range(time_samples):
        e_t = excitation[t]
        # For each MU, calculate firing rate based on excitation
        fr_t = np.minimum(maxfr, minfr + (e_t - rte) * slope_fr)
        fr_t[e_t < rte] = 0  # Zero firing rate if excitation below recruitment threshold
        fr[:, t] = fr_t.flatten()
    
    # Generate spike trains
    spikes = [[] for _ in range(N)]
    next_firing = np.ones(N) * -1
    cur_ipi = np.zeros(N)
    ipi_real = np.zeros((N, time_samples))
    
    # For each motor unit and time point
    for mu in range(N):
        for t in range(time_samples):
            if excitation[t] > rte[mu]:
                if next_firing[mu] < 0:
                    # Initialize next firing time if this is first activation
                    if fr[mu, t] > 0:
                        cur_ipi[mu] = fs / fr[mu, t]
                        # Add variability to IPI
                        cur_ipi[mu] = cur_ipi[mu] + rng.randn() * cur_ipi[mu] * 1/6
                        next_firing[mu] = t + int(cur_ipi[mu])
                
                if t == next_firing[mu]:
                    # Record spike at next firing time
                    if len(spikes[mu]) == 0:
                        ipi_real[mu, :t] = t
                    else:
                        ipi_real[mu, spikes[mu][-1]:t] = t - spikes[mu][-1]
                    
                    spikes[mu].append(t)
                    
                    # Calculate time to next firing
                    if fr[mu, t] > 0:
                        cur_ipi[mu] = fs / fr[mu, t]
                        # Add variability to IPI
                        cur_ipi[mu] = cur_ipi[mu] + rng.randn() * cur_ipi[mu] * 1/6
                        next_firing[mu] = t + int(cur_ipi[mu])
            else:
                next_firing[mu] = -1
    
    # Fill in remaining IPI values
    for mu in range(N):
        if len(spikes[mu]) == 0:
            ipi_real[mu, :] = time_samples
        else:
            ipi_real[mu, spikes[mu][-1]:time_samples] = time_samples - spikes[mu][-1]
    
    return excitation, spikes, fr, ipi_real

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
        if noise_seed is not None:
            np.random.seed(noise_seed)
        
        std_emg = emg.std()
        std_noise = std_emg * 10 ** (-noise_level_db / 20)
        noise = np.random.normal(0, std_noise, emg.shape)
        emg = emg + noise
    
    print(f"EMG generation completed in {time.time() - start_time:.2f} seconds")
    
    return emg

def load_presorted_muaps(muap_file, threshold_file=None):
    """Load pre-sorted MUAPs and optionally their thresholds."""
    try:
        # Load the MUAPs
        print(f"Loading pre-sorted MUAPs from {muap_file}")
        muaps = np.load(muap_file)
        
        # Load thresholds if provided
        thresholds = None
        if threshold_file and os.path.exists(threshold_file):
            print(f"Loading thresholds from {threshold_file}")
            thresholds = np.load(threshold_file)
        
        print(f"Loaded MUAPs with shape: {muaps.shape}")
        if thresholds is not None:
            print(f"Loaded thresholds with shape: {thresholds.shape}")
        
        return muaps, thresholds
    except Exception as e:
        print(f"Error loading pre-sorted MUAPs or thresholds: {e}")
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

def save_outputs(output_dir, emg, spikes, ext, cfg, metadata, selected_thresholds=None):
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
    
    if selected_thresholds is not None:
        paths['thresholds'] = os.path.join(output_dir, f'{subject_prefix}{muscle}_recruitment_thresholds.npz')
        np.savez_compressed(paths['thresholds'], thresholds=selected_thresholds)
    
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

def main():
    parser = argparse.ArgumentParser(description='Generate EMG signals from pre-sorted MUAPs')
    parser.add_argument('config_path', type=str, help='Path to input configuration JSON file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--muap_file', type=str, help='Path to sorted MUAPs file (npy format)')
    parser.add_argument('--threshold_file', type=str, default=None, help='Path to sorted thresholds file (optional)')
    parser.add_argument('--pytorch-device', type=str, choices=['cpu', 'cuda'], default='cpu', help='PyTorch device to use')
    parser.add_argument('--exp-factor', type=float, default=5.0, help='Exponential sampling factor (higher = more bias toward low thresholds)')
    args = parser.parse_args()

    # Set device for PyTorch
    device = args.pytorch_device
    print(f"Using PyTorch device: {device}")
    
    # Check required arguments
    if args.muap_file is None:
        parser.error("--muap_file must be provided")
    
    # Load configuration
    cfg = load_config(args.config_path)
    
    # Subject configuration
    subject_cfg = cfg.SubjectConfiguration
    subject_seed = subject_cfg.SubjectSeed
    subject_id = subject_cfg.get('SubjectID', f"subject_{subject_seed}")
    
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
    
    # Set random seed for reproducibility
    print(f"Using subject seed: {subject_seed}")
    rng = np.random.RandomState(subject_seed)
    torch.manual_seed(subject_seed)
    
    # Load pre-sorted MUAPs and thresholds
    all_muaps, all_thresholds = load_presorted_muaps(args.muap_file, args.threshold_file)
    
    # Determine how many motor units to use (300-350)
    min_mus = 300
    max_mus = 350
    num_mus = rng.randint(min_mus, max_mus + 1)
    
    # Select motor units using exponential sampling
    print(f"Selecting {num_mus} motor units with exponential sampling (factor={args.exp_factor})...")
    selected_muaps, selected_thresholds, selected_indices = exponential_sample_motor_units(
        all_muaps, 
        all_thresholds, 
        num_mus, 
        rng,
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
    
    # Directly generate spike trains using the recruitment thresholds
    print("Generating spike trains based on recruitment thresholds...")
    # Use default settings for firing rate parameters
    settings = {
        'rr': 50,
        'rm': 0.75,
        'rp': 100,
        'pfr1': 40,
        'pfrd': 10,
        'mfr1': 10,
        'mfrd': 5,
        'gain': 30,     # 0.3 per % MVC
        'c_ipi': 0.1,
    }
    
    # Call our direct spike train generation function
    _, spikes, fr, ipis = generate_spike_trains_direct(
        selected_thresholds, 
        effort_profile, 
        fs, 
        settings=settings, 
        rng=rng
    )
    
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
            "movement_duration": movement_duration,
            "movement_dof": movement_cfg.MovementDOF,
            "movement_type": "Isometric",  # Always isometric for this script
            "effort_profile": effort_type,
            "effort_level": profile_params.EffortLevel,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subject_id": subject_id,
            "subject_seed": subject_seed,
            "muap_source": args.muap_file,
            "exponential_sampling_factor": args.exp_factor,
            "selected_indices": selected_indices.tolist()[:10] + ["..."]  # Just show the first 10 indices
        }
    }
    
    # Add threshold range if available
    if selected_thresholds is not None:
        metadata["simulation_info"]["recruitment_threshold_range"] = [
            float(selected_thresholds.min()), 
            float(selected_thresholds.max())
        ]
        metadata["simulation_info"]["recruitment_threshold_quartiles"] = [
            float(np.percentile(selected_thresholds, 25)),
            float(np.percentile(selected_thresholds, 50)),
            float(np.percentile(selected_thresholds, 75))
        ]
    
    # Save all outputs
    save_outputs(
        args.output_dir, 
        emg, 
        spikes, 
        effort_profile, 
        cfg, 
        metadata,
        selected_thresholds
    )
    
    print("EMG generation complete.")

if __name__ == '__main__':
    main()
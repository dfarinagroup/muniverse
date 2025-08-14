#!/usr/bin/env python
"""
_run_neuromotion_new.py - Container script for EMG generation

This script receives input data (movement profiles, MUAPs, config) and returns EMG data.
It handles both neuromotion and hybrid pipelines based on the input data.
"""

import argparse
import json
import os
import numpy as np
import sys
from typing import Dict, List, Tuple
from scipy.signal import butter, filtfilt

# Add current directory to path for imports
sys.path.append(".")

# Container-specific imports (BioMime, NeuroMotion)
try:
    import torch
    from easydict import EasyDict as edict
    from tqdm import tqdm
    from BioMime.models.generator import Generator
    from BioMime.utils.basics import load_generator, update_config
    from NeuroMotion.MNPoollib.mn_params import ANGLE, DEPTH, MS_AREA, NUM_MUS, mn_default_settings
    from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu, normalise_properties
    from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
    from NeuroMotion.MSKlib.MSKpose import MSKModel
except ImportError as e:
    print(f"Warning: Container-specific imports not available: {e}")
    print("This script is intended to run inside a container with BioMime and NeuroMotion")


def generate_emg(muaps, spikes, muap_angle_labels, angle_profile):
    """
    Generate EMG signal based on MUAPs, spikes, muap_angle_labels, and angle_profile.

    Parameters:
    - muaps (ndarray): Array of MUAPs with shape (n_units, steps, ch_rows, ch_cols, win).
    - spikes (list): List of spike timings for each unit.
    - muap_angle_labels (list): List of angle labels for each MUAP.
    - angle_profile (list): List of angles corresponding to each spike timing.

    Returns:
    - emg (ndarray): Generated EMG signal with shape (ch_rows, ch_cols, time_samples).
    """
    # Initialise dimensions
    n_units, steps, ch_rows, ch_cols, win = muaps.shape
    offset = win // 2

    # Initialise emg
    time_samples = len(angle_profile)
    emg = np.zeros((ch_rows, ch_cols, time_samples))

    # Add each unit's contribution
    for unit in range(n_units):
        unit_firings = spikes[unit]

        if len(unit_firings) == 0:
            continue

        for firing in unit_firings:
            # Get the nearest angle's MUAP for each firing
            curr_angle = angle_profile[firing]
            muap_idx = np.argmin(np.abs(muap_angle_labels - curr_angle))
            curr_muap = muaps[unit, muap_idx] 

            # Deal with edge cases
            init_emg = np.max([0, firing - offset])
            end_emg = np.min([firing + offset, time_samples])

            init_muap = init_emg - (firing - offset) # 0 if the window is inside the range
            end_muap = end_emg - (firing + offset) + offset * 2 # win if the window is inside the range

            # Add contribution to EMG
            emg[:, :, init_emg:end_emg] += curr_muap[:, :, init_muap:end_muap]

    return emg


def simulate_dof_range(msk_model, dof, ms_label):
    """
    Simulate the full range of motion for a given degree of freedom and extract muscle parameter changes.
    
    Args:
        movement_cfg: Movement configuration
        ms_label: Target muscle label
        
    Returns:
        Tuple of (changes_dict, num_steps, min_angle, max_angle)
    """
    # Build movement profile for MUAP generation
    fs_mov = 50  # sampling frequency for movement simulation
    if dof == "Flexion-Extension":
        poses = ["ext", "default", "flex"]
        min_angle, max_angle = -65, 65
    elif dof == "Radial-Ulnar-Deviation":
        poses = ["rdev", "default", "udev"]
        min_angle, max_angle = -10, 25
    
    durations = np.abs([min_angle, max_angle]) / fs_mov

    # Simulate movement range
    msk_model.sim_mov(fs_mov, poses, durations)

    # Define muscles to track and extract parameter changes
    ms_labels_msk = ["ECRB", "ECRL", "PL", "FCU", "ECU", "EDCI", "FDSI"]
    _ = msk_model.mov2len(ms_labels=ms_labels_msk) # compute muscle length changes
    d_mu_params = msk_model.len2params() # compute motor unit parameter changes
    num_steps = d_mu_params["steps"]

    # Apply muscle name mapping
    msk_to_biomime_map = {"EDCI": "EDI"}
    for param in ["len", "cv", "depth"]:
        if param in d_mu_params:
            d_mu_params[param] = d_mu_params[param].rename(columns=msk_to_biomime_map)

    return d_mu_params, num_steps, min_angle, max_angle


def initialize_mu_properties(config, mn_pool, num_steps):
    """
    Initialize motor neuron properties and prepare them for MUAP generation.
    
    Args:
        config: Configuration dictionary
        mn_pool: MotoneuronPool object
        num_steps: Number of movement steps
        
    Returns:
        Tuple of (properties_dict, property_tensors, torch_generator)
    """
    # Extract configuration parameters
    subject_cfg = config.get("SubjectConfiguration")
    ms_label = config.get("MovementConfiguration").get("TargetMuscle")
    fibre_density = subject_cfg.get("FibreDensity")
    num_mus = mn_pool.N
    
    # Create dedicated RNG objects for subject seed
    subject_seed = subject_cfg.get("SubjectSeed")
    subject_rng = np.random.RandomState(subject_seed)
    torch_generator = torch.Generator()
    torch_generator.manual_seed(subject_seed)

    # Assign physiological properties using dedicated RNG
    num_fb = np.round(MS_AREA[ms_label] * fibre_density)
    config_props = edict({
        "num_fb": num_fb,
        "depth": DEPTH[ms_label],
        "angle": ANGLE[ms_label],
        "iz": [0.5, 0.1],
        "len": [1.0, 0.05],
        "cv": [4, 0.3],  # Recommend not setting std too large. cv range in training dataset is [3, 4.5]
    })

    # Temporarily set the global state to subject seed for reproducibility
    original_np_state = np.random.get_state()
    np.random.seed(subject_seed)
    properties = mn_pool.assign_properties(config_props, normalise=True)
    np.random.set_state(original_np_state)
    
    properties_dict = {key: val for key, val in properties.items()}

    # Prepare property tensors for all steps
    num = torch.from_numpy(properties["num"]).reshape(num_mus, 1).repeat(1, num_steps)
    depth = torch.from_numpy(properties["depth"]).reshape(num_mus, 1).repeat(1, num_steps)
    angle = torch.from_numpy(properties["angle"]).reshape(num_mus, 1).repeat(1, num_steps)
    iz = torch.from_numpy(properties["iz"]).reshape(num_mus, 1).repeat(1, num_steps)
    cv = torch.from_numpy(properties["cv"]).reshape(num_mus, 1).repeat(1, num_steps)
    length = torch.from_numpy(properties["len"]).reshape(num_mus, 1).repeat(1, num_steps)

    property_tensors = {
        'num': num, 'depth': depth, 'angle': angle, 
        'iz': iz, 'cv': cv, 'length': length
    }

    return properties_dict, property_tensors, torch_generator


def generate_muaps_biomime(config, mu_properties, d_mu_properties, num_steps, ms_label, num_mus, torch_generator):
    """
    Generate MUAPs using BioMime neural network for the full range of motion.
    
    Args:
        config: Configuration dictionary
        mu_properties: Dictionary of motor unit properties
        d_mu_properties: Dictionary of motor unit parameter changes from movement simulation
        num_steps: Number of movement steps
        ms_label: Target muscle label
        num_mus: Number of motor units
        torch_generator: Torch generator for reproducible random number generation
        
    Returns:
        np.ndarray: MUAP library with shape (n_units, steps, ch_rows, ch_cols, win)
    """
    # Extract configuration parameters
    recording_cfg = config.get("RecordingConfiguration")
    fs = recording_cfg.get("SamplingFrequency")
    filter_cfg = recording_cfg.get("FilterProperties")
    
    # Get optional parameters with defaults
    model_pth = config.get("PathToBioMimeWeights", "./ckp/model_linear.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model config and setup generator
    model_config = update_config("./ckp/config.yaml")
    generator = Generator(model_config.Model.Generator)
    generator = load_generator(model_pth, generator, device)
    generator.eval()

    if device == "cuda":
        assert torch.cuda.is_available()
        generator.cuda()

    # Setup filtering parameters
    cutoff = filter_cfg.get("CutoffFrequency")
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = filter_cfg.get("FilterOrder")
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Handle muscle name mapping for BioMime
    # TODO: Handle this outside.
    if ms_label == "FCU_u" or ms_label == "FCU_h":
        tgt_ms_labels = ["FCU"] * num_mus
    else:
        tgt_ms_labels = [ms_label] * num_mus

    ch_depth = d_mu_properties["depth"].loc[:, tgt_ms_labels]
    ch_cv = d_mu_properties["cv"].loc[:, tgt_ms_labels]
    ch_len = d_mu_properties["len"].loc[:, tgt_ms_labels]

    # Generate MUAPs for each movement step
    muaps = []
    for sp in tqdm(range(num_steps), dynamic_ncols=True, desc="Simulating MUAPs during dynamic movement..."):
        # Prepare condition tensor for BioMime
        cond = torch.vstack((
            mu_properties['num'][:, sp],
            mu_properties['depth'][:, sp] * ch_depth.iloc[sp, :].values,
            mu_properties['angle'][:, sp],
            mu_properties['iz'][:, sp],
            mu_properties['cv'][:, sp] * ch_cv.iloc[sp, :].values,
            mu_properties['length'][:, sp] * ch_len.iloc[sp, :].values,
        )).transpose(1, 0)
        
        # Generate MUAPs using BioMime with dedicated generator
        zi = torch.randn(num_mus, model_config.Model.Generator.Latent, generator=torch_generator)
        if device == "cuda":
            zi = zi.cuda()
            cond = cond.cuda()

        sim = generator.sample(num_mus, cond.float(), cond.device, zi)

        if device == "cuda":
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()
        
        # Apply filtering
        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

    # Reshape to final format: (n_units, steps, ch_rows, ch_cols, win)
    muaps = np.array(muaps)
    muaps = np.transpose(muaps, (1, 0, 2, 3, 4))
    
    return muaps


def generate_muap_library(config, mn_pool, msk_model) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate MUAP library using BioMime (neuromotion pipeline), per given dof.
    
    Args:
        config: Configuration dictionary
        mn_pool: MotoneuronPool object for property access
        msk_model: MSKModel object for movement simulation

    Returns:
        Tuple of (muaps, muap_angle_labels, properties_dict)
    """
    # Extract configuration parameters
    movement_cfg = config.get("MovementConfiguration")
    ms_label = movement_cfg.get("TargetMuscle")
    
    # Simulate DOF range and get CV, DEPTH, MS_LEN changes
    d_mu_properties, num_steps, min_angle, max_angle = simulate_dof_range(msk_model, movement_cfg.get("MovementDOF"), ms_label)
    
    # Initialize motor unit properties
    init_mu_properties_dict, mu_properties, torch_generator = initialize_mu_properties(config, mn_pool, num_steps)
    
    # Generate MUAPs using BioMime
    muaps = generate_muaps_biomime(config, mu_properties, d_mu_properties, num_steps, ms_label, mn_pool.N, torch_generator)
    
    # Generate angle labels that correspond to the actual MUAP library
    muap_angle_labels = np.linspace(min_angle, max_angle, num_steps).astype(int)

    return muaps, muap_angle_labels, init_mu_properties_dict


def generate_spike_trains(effort_profile: np.ndarray, mn_pool, fs) -> Tuple[List, np.ndarray]:
    """
    Generate motor unit spike trains.
    
    Args:
        effort_profile: Muscle activation profile
        mn_pool: MotoneuronPool object
        fs: Sampling frequency
        
    Returns:
        Tuple of (spikes_list, firing_rates_array)
    """
    # Initialize the motoneuron pool
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()

    # Generate spike trains
    _, spikes, fr, _ = mn_pool.generate_spike_trains(effort_profile, fit=False)

    return spikes, fr


def main(args):
    """
    Container entry point - receives input data and returns EMG data.
    """
    
    # Use single run directory for all operations
    run_dir = args.run_dir
    
    try:
        # Load config from JSON
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load data arrays from NPZ file
        input_data_path = os.path.join(run_dir, "input_data.npz")
        data = np.load(input_data_path, allow_pickle=True)
        input_data = {key: data[key] for key in data.files}
        
        # Extract data from input
        effort_profile = input_data['effort_profile']
        angle_profile = input_data['angle_profile']
        
    except Exception as e:
        print(f"[ERROR] Failed to load input data from {run_dir}: {e}")
        raise

    # Define a MotorneuronPool object
    ms_label = config.get("MovementConfiguration").get("TargetMuscle")
    
    # Get muscle index from MuscleLabels array
    muscle_labels = config.get("SubjectConfiguration").get("MuscleLabels")
    muscle_mu_counts = config.get("SubjectConfiguration").get("MuscleMotorUnitCounts")
    muscle_index = muscle_labels.index(ms_label)
    num_mus = muscle_mu_counts[muscle_index]

    # TODO: This could be handled in simulate.py or generate_hybrid_recording function
    if ms_label == 'Tibialis_anterior':
        ms_label = 'ECRB' 
        
    mn_pool = MotoneuronPool(num_mus, ms_label, **mn_default_settings)
    
    # Determine MUAP source and get MUAPs
    if 'muaps' in input_data:
        # Hybrid pipeline - use provided MUAPs
        muaps = input_data['muaps']
        muap_angle_labels = input_data['muap_angle_labels']
    else:
        # Neuromotion pipeline - generate MUAP library, look up angle labels
        msk_model = MSKModel(
            model_path="./NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3/",
            model_name="Hand_Wrist_Model_for_development.osim",
            default_pose_path="./NeuroMotion/MSKlib/models/poses.csv",
        )
        muaps, muap_angle_labels, properties = generate_muap_library(config, mn_pool, msk_model)
    
    # Generate spike trains
    fs = config.get("RecordingConfiguration").get("SamplingFrequency")
    spikes, firing_rates = generate_spike_trains(effort_profile, mn_pool, fs)
    
    # Generate EMG signal
    emg = generate_emg(muaps, spikes, muap_angle_labels, angle_profile)
    
    # Prepare output data
    output_data = {
        'emg': emg,
        'spikes': spikes,
        'firing_rates': firing_rates,
        'effort_profile': effort_profile,
        'angle_profile': angle_profile,
        'muaps': muaps,
        'muap_angle_labels': muap_angle_labels,
        'properties': properties,
        'config': config,
    }

    # Save output data in the same run directory
    output_path = os.path.join(run_dir, "emg_data.npz")
    np.savez_compressed(output_path, **output_data)
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EMG signals from input data")
    parser.add_argument("--run_dir", type=str, required=True, 
                       help="Path to run directory containing input files and where output will be saved")
    
    args = parser.parse_args()
    main(args) 
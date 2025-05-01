#!/usr/bin/env python3
# run_neuromotion.py - Enhanced version that uses subject-specific configuration

import argparse
import os
import torch
import time
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt

import sys
sys.path.append('.')

from NeuroMotion.MSKlib.MSKpose import MSKModel
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu, normalise_properties
from NeuroMotion.MNPoollib.mn_params import DEPTH, ANGLE, MS_AREA, NUM_MUS, mn_default_settings
from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:  # Fallback for YAML if needed
            import yaml
            config = yaml.safe_load(f)
    return edict(config)


# Subject configuration is now integrated into the main config file


def build_movement_profile(movement_config):
    """
    Build a movement profile based on the configuration.
    Always generates the full range of motion for the chosen degree of freedom
    to create a complete library of MUAPs.
    
    Args:
        movement_config: Movement configuration from the JSON file
    
    Returns:
        tuple: (poses, durations, total_duration, steps)
    """
    movement_dof = movement_config.MovementDOF
    
    # For MUAP generation, we always use the full range of motion
    fs_mov = 50  # temporal frequency in Hz (sampling rate for movement)
    
    if movement_dof == "Flexion-Extension":
        # For flexion-extension, we go from max extension to max flexion
        # Default range is -65 (extension) to 65 (flexion) degrees
        r_ext = -65
        r_flex = 65
        
        # Always use the full range of motion for the MUAP library
        poses = ['ext', 'default', 'flex']
        durations = np.abs([r_ext, r_flex]) / fs_mov
        
    elif movement_dof == "Radial-Ulnar-deviation":
        # For radial-ulnar deviation, default values
        r_rad = -10
        r_uln = 25
        
        # Always use the full range of motion for the MUAP library
        poses = ['rdev', 'default', 'udev']
        durations = np.abs([r_rad, r_uln]) / fs_mov
        

    else:
        raise ValueError(f"Unsupported MovementDOF: '{movement_dof}'. Only 'Flexion-Extension' and 'Radial-Ulnar-deviation' are supported.")

    # Calculate the total duration and number of steps
    total_duration = np.sum(durations)
    steps = int(np.round(total_duration * fs_mov))
    
    return poses, durations, total_duration, steps


def create_trapezoid_effort(fs, movement_duration, effort_level, rest_duration, ramp_duration, hold_duration, n_reps):
    """Create a trapezoidal effort profile.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        effort_level (float): Maximum effort level (0-1).
        rest_duration (float): Duration of rest period in seconds.
        ramp_duration (float): Duration of ramp up/down in seconds.
        hold_duration (float): Duration of sustained effort in seconds.
        n_reps (int): Number of repetitions.
        
    Returns:
        numpy.ndarray: Effort profile.
    """
    # Create the effort profile
    ext = np.zeros(round(fs * movement_duration))
    current_idx = 0
    
    for rep in range(n_reps):
        # Rest period
        if current_idx + round(fs * rest_duration) <= len(ext):
            current_idx += round(fs * rest_duration)
        else:
            break
        
        # Ramp up
        ramp_up = np.linspace(0, effort_level, round(fs * ramp_duration))
        if current_idx + len(ramp_up) <= len(ext):
            ext[current_idx:current_idx+len(ramp_up)] = ramp_up
            current_idx += len(ramp_up)
        else:
            break
        
        # Hold
        if current_idx + round(fs * hold_duration) <= len(ext):
            ext[current_idx:current_idx+round(fs * hold_duration)] = effort_level
            current_idx += round(fs * hold_duration)
        else:
            # Fill remaining with effort_level
            remaining = len(ext) - current_idx
            if remaining > 0:
                ext[current_idx:] = effort_level
            break
        
        # Ramp down
        ramp_down = np.linspace(effort_level, 0, round(fs * ramp_duration))
        if current_idx + len(ramp_down) <= len(ext):
            ext[current_idx:current_idx+len(ramp_down)] = ramp_down
            current_idx += len(ramp_down)
        else:
            # If we can't fit a full ramp down, fill with a partial ramp
            remaining = len(ext) - current_idx
            if remaining > 0:
                partial_ramp = np.linspace(effort_level, effort_level * (1 - remaining/len(ramp_down)), remaining)
                ext[current_idx:] = partial_ramp
            break
    
    return ext


def create_triangular_effort(fs, movement_duration, effort_level, rest_duration, ramp_duration, n_reps=1):
    """Create a triangular effort profile with specified parameters.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        effort_level (float): Maximum effort level (0-1).
        rest_duration (float): Duration of rest period in seconds.
        ramp_duration (float): Duration of ramp up/down in seconds.
        n_reps (int, optional): Number of repetitions. Defaults to 1.
        
    Returns:
        numpy.ndarray: Effort profile.
    """
    # Create the effort profile
    ext = np.zeros(round(fs * movement_duration))
    current_idx = 0
    
    for rep in range(n_reps):
        # Rest period
        if current_idx + round(fs * rest_duration) <= len(ext):
            current_idx += round(fs * rest_duration)
        else:
            break
        
        # Calculate maximum possible triangle duration
        max_triangle_duration = len(ext) - current_idx
        triangle_duration = round(fs * ramp_duration * 2)  # Up and down
        
        if triangle_duration > max_triangle_duration:
            # Adjust ramp duration to fit in the remaining time
            adjusted_ramp = max_triangle_duration / (2 * fs)
            ramp_samples = round(max_triangle_duration / 2)
        else:
            ramp_samples = round(fs * ramp_duration)
        
        # Ramp up
        if current_idx + ramp_samples <= len(ext):
            ramp_up = np.linspace(0, effort_level, ramp_samples)
            ext[current_idx:current_idx+ramp_samples] = ramp_up
            current_idx += ramp_samples
        else:
            break
        
        # Ramp down
        if current_idx + ramp_samples <= len(ext):
            ramp_down = np.linspace(effort_level, 0, ramp_samples)
            ext[current_idx:current_idx+ramp_samples] = ramp_down
            current_idx += ramp_samples
        else:
            # If we can't fit a full ramp down, fill with a partial ramp
            remaining = len(ext) - current_idx
            if remaining > 0:
                partial_ramp = np.linspace(effort_level, effort_level * (1 - remaining/ramp_samples), remaining)
                ext[current_idx:current_idx+remaining] = partial_ramp
            break
    
    return ext


def create_ballistic_effort(fs, movement_duration, effort_level, rest_duration, n_reps=1, ramp_duration=0.05):
    """Create a ballistic effort profile with very quick ramp up.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        effort_level (float): Maximum effort level (0-1).
        rest_duration (float): Duration of rest period in seconds.
        n_reps (int, optional): Number of repetitions. Defaults to 1.
        ramp_duration (float, optional): Duration of ramp up in seconds. Defaults to 0.05s (50ms).
        
    Returns:
        numpy.ndarray: Effort profile.
    """
    # For ballistic contractions, we use a very short ramp up (default 50ms)
    # followed by immediate return to rest
    
    # Create one contraction cycle
    rest_samples = round(fs * rest_duration)
    ramp_samples = round(fs * ramp_duration)
    
    # One contraction consists of rest followed by rapid ramp up
    one_contraction = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, effort_level, ramp_samples)
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


def create_sinusoidal_effort(fs, movement_duration, effort_level, sin_frequency, sin_amplitude=None, rest_duration=0):
    """Create a sinusoidal effort profile with optional base effort level and amplitude.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        effort_level (float): Base effort level (0-1).
        sin_frequency (float): Frequency of sinusoid in Hz.
        sin_amplitude (float, optional): Amplitude of sinusoid oscillation (0-1).
            If None, uses half of the effort_level.
        rest_duration (float, optional): Initial rest period in seconds.
        
    Returns:
        numpy.ndarray: Effort profile.
    """
    # Convert amplitude from percentage to decimal if provided
    if sin_amplitude is not None:
        sin_amplitude = sin_amplitude / 100.0
    else:
        sin_amplitude = effort_level / 2.0
    
    # Time vector (excluding rest period)
    active_duration = movement_duration - rest_duration
    t = np.arange(round(fs * active_duration)) / fs
    
    # Create sinusoidal profile
    # Ensure effort level stays within [0, 1] range
    # Base oscillation around effort_level with amplitude sin_amplitude
    active_profile = effort_level + sin_amplitude * np.sin(2 * np.pi * sin_frequency * t)
    
    # Clip to ensure values stay in valid range [0, 1]
    active_profile = np.clip(active_profile, 0, 1)
    
    # Add rest period if specified
    if rest_duration > 0:
        rest_samples = round(fs * rest_duration)
        rest_profile = np.zeros(rest_samples)
        ext = np.concatenate((rest_profile, active_profile))
    else:
        ext = active_profile
    
    # Ensure the total length matches the movement duration
    expected_length = round(fs * movement_duration)
    if len(ext) < expected_length:
        ext = np.pad(ext, (0, expected_length - len(ext)), 'constant')
    elif len(ext) > expected_length:
        ext = ext[:expected_length]
        
    return ext


def create_default_effort(fs, movement_duration, effort_level):
    """Create a default effort profile with ramp up, hold, and ramp down.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        effort_level (float): Maximum effort level (0-1).
        
    Returns:
        numpy.ndarray: Effort profile.
    """
    # Default is a simple ramp up, hold, ramp down pattern
    ext = np.concatenate((
        np.linspace(0, effort_level, round(fs * movement_duration / 3)), 
        np.ones(round(fs * movement_duration / 3)) * effort_level,
        np.linspace(effort_level, 0, round(fs * movement_duration / 3))
    ))
    
    # Pad if needed
    if len(ext) < round(fs * movement_duration):
        ext = np.pad(ext, (0, round(fs * movement_duration) - len(ext)), 'constant')
    
    return ext


def create_constant_effort(fs, movement_duration, effort_level):
    """
    Create an effort profile that stays at one level for the entire duration.

    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        effort_level (float): Target effort level (0–1).

    Returns:
        numpy.ndarray: 1-D array of length fs*movement_duration, filled with effort_level.
    """
    return np.full(round(fs * movement_duration), effort_level, dtype=float)


def create_effort_profile(fs, movement_duration, profile_params):
    """Create an effort profile based on the movement parameters.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        profile_params (easydict.EasyDict): Movement profile parameters.
        
    Returns:
        numpy.ndarray: Effort profile.
    """
    effort_level = profile_params.EffortLevel / 100.0  # Convert percentage to decimal
    
    if hasattr(profile_params, 'EffortProfile'):
        if profile_params.EffortProfile == "Trapezoid":
            return create_trapezoid_effort(
                fs, 
                movement_duration, 
                effort_level, 
                profile_params.RestDuration, 
                profile_params.RampDuration, 
                profile_params.HoldDuration, 
                profile_params.NRepetitions
            )
        elif profile_params.EffortProfile == "Sinusoid":
            # Convert sinusoidal parameters from percentages to decimal
            sin_amplitude = getattr(profile_params, 'SinAmplitude', None)
            rest_duration = getattr(profile_params, 'RestDuration', 0)
            return create_sinusoidal_effort(
                fs, 
                movement_duration, 
                effort_level, 
                profile_params.SinFrequency,
                sin_amplitude,
                rest_duration
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
        elif profile_params.EffortProfile == "Ballistic":
            n_reps = getattr(profile_params, 'NRepetitions', 1)
            ramp_duration = getattr(profile_params, 'RampDuration', 0.05)  # Default 50ms
            return create_ballistic_effort(
                fs,
                movement_duration,
                effort_level,
                profile_params.RestDuration,
                n_reps,
                ramp_duration
            )
        elif profile_params.EffortProfile == "Constant":
            return create_constant_effort(
                fs,
                movement_duration,
                effort_level
            )

    
    # Default case
    return create_default_effort(fs, movement_duration, effort_level)


def generate_angle_profile(fs, movement_duration, profile_params, movement_dof, muap_dof_samples):
    """Create an angle profile based on the movement parameters.
    
    Args:
        fs (float): Sampling frequency in Hz.
        movement_duration (float): Total duration in seconds.
        profile_params (easydict.EasyDict): Movement profile parameters.
        movement_dof (str): Degree of freedom ("Flexion-Extension" or "Radial-Ulnar").
        muap_dof_samples (int): Number of MUAP samples for the given DoF
        
    Returns:
        tuple: (angle_profile, muap_angle_labels)
            - angle_profile: Array of angles for each time point
            - muap_angle_labels: Array of angles corresponding to the MUAP library
    """
    # Generate angle labels for the MUAP library based on the DOF
    if movement_dof == "Flexion-Extension":
        # Default range for Flexion-Extension: -65 (extension) to 65 (flexion) degrees
        min_angle = -65
        max_angle = 65
    elif movement_dof == "Radial-Ulnar":
        # Default range for radial-ulnar deviation: -10 (radial) to 25 (ulnar) degrees
        min_angle = -10
        max_angle = 25
    else:
        # Default range for unknown DOF
        min_angle = -60
        max_angle = 60
    
    # Create the angle labels that correspond to the MUAP library
    # Assuming 130 angle steps in the MUAP library (can be adjusted)
    num_angle_steps = muap_dof_samples
    muap_angle_labels = np.linspace(min_angle, max_angle, num_angle_steps).astype(int)
    
    # Create angle profile based on the specified profile type
    if hasattr(profile_params, 'AngleProfile'):
        if profile_params.AngleProfile == "Constant":
            # Constant angle (isometric) - the simplest case
            target_angle = getattr(profile_params, 'TargetAngle', 0)
            # Ensure the target angle is within the range of available angles
            target_angle = np.clip(target_angle, min_angle, max_angle)
            angle_profile = np.ones(round(fs * movement_duration)) * target_angle
        
        elif profile_params.AngleProfile == "Sinusoid":
            # Sinusoidal angle variation
            target_angle_percentage = getattr(profile_params, 'TargetAnglePercentage', 0.75)
            target_angle_direction = getattr(profile_params, 'TargetAngleDirection', 1)
            sin_amplitude_ratio = getattr(profile_params, 'SinAmplitude', 0.3)
            sin_frequency = getattr(profile_params, 'SinFrequency', 0.2)

            # Calculate full range
            angle_range = max_angle - min_angle

            # Determine base (target) angle
            if target_angle_direction == 0:
                target_angle = min_angle + angle_range * target_angle_percentage
            else:
                target_angle = max_angle - angle_range * target_angle_percentage

            # Calculate absolute sinusoidal amplitude (in degrees/radians)
            sin_amplitude = angle_range * sin_amplitude_ratio

            # Time vector
            t = np.arange(round(fs * movement_duration)) / fs

            # Generate sinusoidal angle profile around the target angle
            angle_profile = target_angle + sin_amplitude * np.sin(2 * np.pi * sin_frequency * t)

            # Clip profile to stay within the physical limits
            angle_profile = np.clip(angle_profile, min_angle, max_angle)
        elif profile_params.AngleProfile == "Ramp":
            # Ramp angle variation (linear change between start and end angles)
            start_angle = getattr(profile_params, 'StartAngle', min_angle)
            end_angle = getattr(profile_params, 'EndAngle', max_angle)
            
            # Create linear ramp
            angle_profile = np.linspace(start_angle, end_angle, round(fs * movement_duration))
                
        elif profile_params.AngleProfile == "Triangular":
            target_angle_percentage = getattr(profile_params, 'TargetAnglePercentage', 0.8)
            target_angle_direction = getattr(profile_params, 'TargetAngleDirection', 1)
            ramp_duration = getattr(profile_params, 'RampDuration', 5)  # in seconds
            n_repetitions = getattr(profile_params, 'NRepetitions', 3)

            # Total number of samples
            total_samples = round(fs * movement_duration)
            
            # Compute angle range and target angle
            angle_range = max_angle - min_angle
            if target_angle_direction == 0:
                target_angle = min_angle + angle_range * target_angle_percentage
            else:
                target_angle = max_angle - angle_range * target_angle_percentage

            # One ramp: up and down (triangle wave), duration is 2 * ramp_duration
            samples_per_cycle = round(fs * 2 * ramp_duration)
            
            # Generate one cycle
            up_ramp = np.linspace(min_angle, target_angle, samples_per_cycle // 2)
            down_ramp = np.linspace(target_angle, min_angle, samples_per_cycle // 2)
            single_cycle = np.concatenate([up_ramp, down_ramp])
            
            # Repeat the cycle
            angle_profile = np.tile(single_cycle, n_repetitions)
            
            # Clip or pad to fit total duration
            if len(angle_profile) < total_samples:
                pad_len = total_samples - len(angle_profile)
                angle_profile = np.pad(angle_profile, (0, pad_len), mode='edge')
            else:
                angle_profile = angle_profile[:total_samples]
        else:
            # Default to constant profile at 0 degrees if profile type not recognized
            angle_profile = np.ones(round(fs * movement_duration)) * 0
    else:
        # Default to constant profile at 0 degrees if no profile specified
        angle_profile = np.ones(round(fs * movement_duration)) * 0
    
    return angle_profile, muap_angle_labels


def get_curr_angle_muap(curr_angle, unit_muaps_angles, angle_labels):
    """Get the MUAP corresponding to the current angle

    Args:
        curr_angle (float): current angle
        unit_muaps_angles (np.ndarray): Array of MUAPs with shape (units, morphs, ch_rows, ch_cols, samples)
        angle_labels (np.ndarray): Range of generated angles

    Returns:
        np.ndarray: MUAPs corresponding to the current angle
    """    

    idx = np.argmin( np.abs(angle_labels - curr_angle) )
    muap = unit_muaps_angles[idx]

    return muap


def generate_emg(muaps, spikes, muap_angle_labels, angle_profile):
    """
    Generate EMG signal based on MUAPs, spikes, muap_angle_labels, and angle_profile.

    Parameters:
    - muaps (ndarray): Array of MUAPs with shape (_, _, ch_rows, ch_cols, win).
    - spikes (list): List of spike timings for each unit.
    - muap_angle_labels (list): List of angle labels for each MUAP.
    - angle_profile (list): List of angles corresponding to each spike timing.

    Returns:
    - emg (ndarray): Generated EMG signal with shape (ch_rows, ch_cols, time_samples).
    """

    # Initialise dimensions
    _, _, ch_rows, ch_cols, win = muaps.shape
    offset = win//2
    
    # Check number of active units
    units_active = 0
    for sp in spikes:
        if len(sp) > 0:
            units_active += 1

    # Initialise emg
    time_samples = len(angle_profile)
    emg = np.zeros((ch_rows, ch_cols, time_samples))

    # Add each unit's contribution
    for unit in range(units_active):
        unit_firings = spikes[unit]

        if len(unit_firings) == 0:
            continue

        for firing in unit_firings:
            # Get the corresponding MUAP morphing for each firing
            curr_angle = angle_profile[firing]
            curr_muap = get_curr_angle_muap(curr_angle, muaps[unit], muap_angle_labels)

            # Deal with edge cases
            init_emg = np.max([0, firing-offset])
            end_emg = np.min([firing+offset, time_samples])

            init_muap = init_emg - (firing-offset)          # 0 if the window is inside the range
            end_muap = end_emg - (firing+offset) + offset*2   # win if the window is inside the range

            # Add contribution to EMG
            emg[:,:, init_emg:end_emg] += curr_muap[:,:,init_muap:end_muap]

    return emg


def generate_spike_trains(mn_pool, effort_profile, fs):
    """Generate spike trains based on an effort profile.
    
    Args:
        mn_pool (MotoneuronPool): The motoneuron pool object.
        effort_profile (numpy.ndarray): The effort profile.
        fs (float): Sampling frequency in Hz.
    
    Returns:
        tuple: (modified_effort, spikes, firing_rates, inter_pulse_intervals)
    """
    # Initialize the motoneuron pool
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()
    
    # Generate spike trains
    ext_new, spikes, fr, ipis = mn_pool.generate_spike_trains(effort_profile, fit=False)
    
    return ext_new, spikes, fr, ipis


def generate_emg_signal(muaps, spikes, time_samples, muap_angle_labels, angle_profile, noise_level_db=None, noise_seed=None):
    """Generate EMG signal by convolving MUAPs with spike trains, using angle profiles.
    
    Args:
        muaps (numpy.ndarray): MUAPs with shape (num_mus, angle_steps, n_row, n_col, time_length).
        spikes (list): List of spike trains.
        time_samples (int): Number of time samples in the effort profile.
        muap_angle_labels (numpy.ndarray): Array of angles corresponding to the MUAP library.
        angle_profile (numpy.ndarray): Angle profile for the simulation.
        noise_level_db (float, optional): Noise level in dB. If None, no noise is added.
        noise_seed (int, optional): Random seed for noise generation.
    
    Returns:
        numpy.ndarray: EMG signal with shape (samples, n_row*n_col).
    """
    start_time = time.time()
    num_mus = len(spikes)
    
    # Use the generate_emg function from NeuroMotion library for correct angle-based MUAP selection
    #from NeuroMotion.MNPoollib.mn_utils import generate_emg
    emg_raw = generate_emg(muaps, spikes, muap_angle_labels, angle_profile)
    
    # Reshape to match expected format (samples, channels)
    _, _, n_row, n_col, _ = muaps.shape
    chs = n_row * n_col
    emg_raw = emg_raw.reshape(chs, -1).T  # Transpose to get (samples, channels)
    
    # Add noise if specified
    if noise_level_db is not None:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        
        std_emg = emg_raw.std()
        std_noise = std_emg * 10 ** (-noise_level_db / 20)
        noise = np.random.normal(0, std_noise, emg_raw.shape)
        emg_raw = emg_raw + noise
    
    print(f"EMG generation completed in {time.time() - start_time:.2f} seconds")
    
    return emg_raw


def select_optimal_electrode_columns(emg, n_rows, n_cols, desired_cols):
    """
    Selects the optimal subset of electrode columns based on signal energy.
    
    Args:
        emg (numpy.ndarray): EMG signal with shape (samples, n_rows*n_cols).
        n_rows (int): Number of rows in the electrode array.
        n_cols (int): Number of columns in the electrode array.
        desired_cols (int): Number of columns to select.
        
    Returns:
        tuple: (selected_emg, column_indices, center_column)
            - selected_emg: EMG signal with only the selected columns
            - column_indices: Indices of the selected columns
            - center_column: Index of the column with highest RMS
    """
    # Ensure desired columns is not larger than available columns
    if desired_cols >= n_cols:
        print(f"Requested {desired_cols} columns, but only {n_cols} available. Using all columns.")
        return emg, list(range(n_cols)), n_cols // 2
    
    # Reshape to separate rows and columns
    emg_reshaped = emg.reshape(-1, n_rows, n_cols)
    
    # Calculate RMS per column (averaging across all rows)
    rms_per_column = np.sqrt(np.mean(np.square(emg_reshaped), axis=(0, 1)))
    
    # Find the column with the highest RMS
    max_col_idx = int(np.argmax(rms_per_column))
    print(f"Column with highest RMS: {max_col_idx}")
    
    # Calculate how many columns to take on each side of the max column
    half_width = desired_cols // 2
    
    # Handle edge cases
    if max_col_idx - half_width < 0:
        # Too clos e to left edge, shift window right
        start_col = 0
        end_col = desired_cols
    elif max_col_idx + half_width >= n_cols:
        # Too close to right edge, shift window left
        start_col = n_cols - desired_cols
        end_col = n_cols
    else:
        # Centered around max column
        start_col = max_col_idx - half_width
        end_col = start_col + desired_cols
    
    # Select the columns
    selected_cols = list(range(start_col, end_col))
    
    # Reshape EMG and select only the desired columns
    selected_indices = []
    for col in selected_cols:
        selected_indices.extend([col * n_rows + row for row in range(n_rows)])
    
    selected_emg = emg[:, selected_indices]
    
    print(f"Selected columns {start_col} to {end_col-1} (centered around column {max_col_idx})")
    
    return selected_emg, selected_cols, max_col_idx


def generate_muaps(
    model_pth, ms_label, movement_cfg, fs_mov, poses, durations, steps, 
    device, morph, muap_file, fibre_density, fs, filter_cfg, n_rows, n_cols, 
    num_mus, subject_seed=None
):
    """Generate MUAPs for the full range of motion.
    
    Args:
        model_pth (str): Path to the model weights.
        ms_label (str): Target muscle label.
        movement_cfg (easydict.EasyDict): Movement configuration.
        fs_mov (float): Movement sampling frequency.
        poses (list): List of poses.
        durations (list): List of durations.
        steps (int): Number of steps.
        device (str): Device to use ('cuda' or 'cpu').
        morph (bool): Whether to morph MUAPs.
        muap_file (str): Path to MUAP file for morphing.
        fibre_density (float): Fibre density.
        fs (float): Sampling frequency.
        filter_cfg (easydict.EasyDict): Filter configuration.
        n_rows (int): Number of rows in electrode array.
        n_cols (int): Number of columns in electrode array.
        num_mus (int): Number of motor units to generate.
        subject_seed (int, optional): Subject-specific random seed.
        
    Returns:
        tuple: (muaps, num_mus)
    """
    # Set subject-specific seed if provided
    if subject_seed is not None:
        original_np_state = np.random.get_state()
        original_torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            original_cuda_state = torch.cuda.get_rng_state()
        
        np.random.seed(subject_seed)
        torch.manual_seed(subject_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(subject_seed)
    
    # Load model config
    model_config = update_config('./ckp/config.yaml')
    
    # PART ONE: Define MSK model, movements, and extract param changes
    msk = MSKModel(
        model_path = './NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3/',
        model_name= 'Hand_Wrist_Model_for_development.osim',
        default_pose_path = './NeuroMotion/MSKlib/models/poses.csv',
    )
    
    # Simulate movement
    msk.sim_mov(fs_mov, poses, durations)

    # Define muscles to track
    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']
    ms_lens = msk.mov2len(ms_labels=ms_labels)
    changes = msk.len2params()
    steps = changes['steps']

    # PART TWO: Define the MotoneuronPool of one muscle
    if morph:
        with open(muap_file, 'rb') as fl:
            db = pickle.load(fl)
        
    # Use specified number of motor units
    mn_pool = MotoneuronPool(num_mus, ms_label, **mn_default_settings)
    
# Create a dictionary to store the properties
    properties_dict = {}
    
    # Assign physiological properties
    num_fb = np.round(MS_AREA[ms_label] * fibre_density)    # total number within one muscle
    config = edict({
        'num_fb': num_fb,
        'depth': DEPTH[ms_label],
        'angle': ANGLE[ms_label],
        'iz': [0.5, 0.1],
        'len': [1.0, 0.05],
        'cv': [4, 0.3]      # Recommend not setting std too large. cv range in training dataset is [3, 4.5]
    })

    if morph:
        num, depth, angle, iz, cv, length, base_muaps = normalise_properties(db, num_mus, steps)
        # Store properties in dictionary
        properties_dict = {
            'num': num.cpu().numpy() if isinstance(num, torch.Tensor) else num,
            'depth': depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth,
            'angle': angle.cpu().numpy() if isinstance(angle, torch.Tensor) else angle,
            'iz': iz.cpu().numpy() if isinstance(iz, torch.Tensor) else iz,
            'cv': cv.cpu().numpy() if isinstance(cv, torch.Tensor) else cv,
            'len': length.cpu().numpy() if isinstance(length, torch.Tensor) else length
        }
    else:
        properties = mn_pool.assign_properties(config, normalise=True)
        # Store original properties
        properties_dict = {key: val for key, val in properties.items()}
        
        # Continue with existing code
        num = torch.from_numpy(properties['num']).reshape(num_mus, 1).repeat(1, steps)
        depth = torch.from_numpy(properties['depth']).reshape(num_mus, 1).repeat(1, steps)
        angle = torch.from_numpy(properties['angle']).reshape(num_mus, 1).repeat(1, steps)
        iz = torch.from_numpy(properties['iz']).reshape(num_mus, 1).repeat(1, steps)
        cv = torch.from_numpy(properties['cv']).reshape(num_mus, 1).repeat(1, steps)
        length = torch.from_numpy(properties['len']).reshape(num_mus, 1).repeat(1, steps)
    
    # PART THREE: Simulate MUAPs using BioMime during the movement
    if ms_label == 'FCU_u' or ms_label == 'FCU_h':
        tgt_ms_labels = ['FCU'] * num_mus
    else:
        tgt_ms_labels = [ms_label] * num_mus

    ch_depth = changes['depth'].loc[:, tgt_ms_labels]
    ch_cv = changes['cv'].loc[:, tgt_ms_labels]
    ch_len = changes['len'].loc[:, tgt_ms_labels]
    

    # Model
    generator = Generator(model_config.Model.Generator)
    generator = load_generator(model_pth, generator, device)
    generator.eval()

    # Device
    if device == 'cuda':
        assert torch.cuda.is_available()
        generator.cuda()

    # Filtering parameters from config
    time_length = 96  # Default time length for MUAP
    cutoff = filter_cfg.CutoffFrequency
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = filter_cfg.FilterOrder
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    print(f"Starting simulation with {num_mus} motor units, {steps} movement steps")
    start_time = time.time()

    muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True, desc='Simulating MUAPs during dynamic movement...'):
        cond = torch.vstack((
            num[:, sp],
            depth[:, sp] * ch_depth.iloc[sp, :].values,
            angle[:, sp],
            iz[:, sp],
            cv[:, sp] * ch_cv.iloc[sp, :].values,
            length[:, sp] * ch_len.iloc[sp, :].values,
        )).transpose(1, 0)
        if not morph:
            zi = torch.randn(num_mus, model_config.Model.Generator.Latent)
            if device == 'cuda':
                zi = zi.cuda()
        else:
            if device == 'cuda':
                base_muaps = base_muaps.cuda()

        if device == 'cuda':
            cond = cond.cuda()

        if morph:
            sim = generator.generate(base_muaps, cond.float())
        else:
            sim = generator.sample(num_mus, cond.float(), cond.device, zi)

        if device == 'cuda':
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()
        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

    muaps = np.array(muaps)
    muaps = np.transpose(muaps, (1, 0, 2, 3, 4))
    print('--- %s seconds to generate MUAPs ---' % (time.time() - start_time))
    
    # Restore original random states if using subject-specific seed
    if subject_seed is not None:
        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(original_cuda_state)
    
    # Return the properties along with the muaps and num_mus
    return muaps, num_mus, properties_dict

def save_motor_unit_properties_to_csv(properties_dict, csv_file):
    """
    Convert motor unit properties to a DataFrame and save as CSV.
    
    Args:
        properties_dict (dict): Dictionary with property names as keys and numpy arrays as values.
            Each array should have shape (num_mus,) or (num_mus, steps).
        csv_file (str): Path to save the CSV file.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    import pandas as pd
    
    try:
        # Create a DataFrame with one row per motor unit
        df = pd.DataFrame()
        
        # For each property, get the first value (assuming static or just want initial value)
        for prop_name, prop_values in properties_dict.items():
            # If property has shape (num_mus, steps), take the first step
            if isinstance(prop_values, torch.Tensor):
                prop_values = prop_values.cpu().numpy()
                
            if len(prop_values.shape) > 1 and prop_values.shape[1] > 1:
                df[prop_name] = prop_values[:, 0]
            else:
                df[prop_name] = prop_values
        
        # Add a motor unit index column
        df.insert(0, 'motor_unit_index', range(len(df)))
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        print(f"Saved motor unit properties to {csv_file}")
        return True
    except Exception as e:
        print(f"Error saving motor unit properties: {e}")
        return False


def cache_muaps(muaps, cache_file, metadata, meta_file, properties_dict=None, properties_csv=None):
    """Cache MUAPs and metadata to files.
    
    Args:
        muaps (numpy.ndarray): MUAPs to cache.
        cache_file (str): Path to save the MUAPs.
        metadata (dict): Metadata about the MUAPs.
        meta_file (str): Path to save the metadata.
        properties_dict (dict, optional): Dictionary of motor unit properties.
        properties_csv (str, optional): Path to save properties CSV.
        
    Returns:
        bool: True if caching succeeded, False otherwise.
    """
    try:
        np.save(cache_file, muaps)
        
        # Save properties if provided
        if properties_dict is not None and properties_csv is not None:
            save_motor_unit_properties_to_csv(properties_dict, properties_csv)
            # Add properties CSV path to metadata
            metadata['properties_csv'] = properties_csv
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved MUAP cache to {cache_file}")
        return True
    except Exception as e:
        print(f"Error saving MUAP cache: {e}")
        return False

def load_cached_muaps(cache_file, meta_file, required_metadata):
    """Load cached MUAPs if they exist and are compatible.
    
    Args:
        cache_file (str): Path to the cached MUAPs.
        meta_file (str): Path to the metadata file.
        required_metadata (dict): Required metadata for compatibility.
        
    Returns:
        tuple: (muaps, is_compatible, num_mus, properties_dict)
            - muaps: The loaded MUAPs or None if loading failed
            - is_compatible: True if the cache is compatible, False otherwise
            - num_mus: Number of motor units in the cache
            - properties_dict: Dictionary of motor unit properties or None
    """
    properties_dict = None
    
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        try:
            # Load metadata to check if the cache matches our requirements
            with open(meta_file, 'r') as f:
                muap_metadata = json.load(f)
            
            # Check if the cache is compatible
            cache_compatible = (
                muap_metadata.get('muscle') == required_metadata['muscle'] and
                muap_metadata.get('fs') == required_metadata['fs'] and
                muap_metadata.get('electrode_rows') == required_metadata['electrode_rows'] and
                muap_metadata.get('electrode_cols') == required_metadata['electrode_cols'] and
                muap_metadata.get('subject_id') == required_metadata.get('subject_id')
            )
            
            if cache_compatible:
                print(f"Found compatible MUAP cache for subject {required_metadata.get('subject_id')}, muscle {required_metadata['muscle']}, loading...")
                muaps = np.load(cache_file, allow_pickle=True)
                num_mus = muap_metadata.get('num_mus')
                
                # Load properties if available
                properties_csv = muap_metadata.get('properties_csv')
                if properties_csv and os.path.exists(properties_csv):
                    import pandas as pd
                    try:
                        df = pd.read_csv(properties_csv)
                        # Convert DataFrame back to dictionary of numpy arrays
                        properties_dict = {col: df[col].values for col in df.columns if col != 'motor_unit_index'}
                        print(f"Loaded motor unit properties from {properties_csv}")
                    except Exception as e:
                        print(f"Error loading properties CSV: {e}")
                
                print(f"Loaded cached MUAPs from {cache_file}, shape: {muaps.shape}")
                return muaps, True, num_mus, properties_dict
        except Exception as e:
            print(f"Error loading MUAP cache: {e}")
    
    return None, False, None, None


def save_outputs(output_dir, emg, spikes, ext, cfg, metadata, angle_profile=None, muaps=None, properties_dict=None):
    """Save all outputs to the specified directory.
    
    Args:
        output_dir (str): Output directory.
        emg (numpy.ndarray): EMG signal.
        spikes (list): List of spike trains.
        ext (numpy.ndarray): Effort profile.
        cfg (easydict.EasyDict): Configuration.
        metadata (dict): Metadata.
        angle_profile (numpy.ndarray, optional): Angle profile.
        muaps (numpy.ndarray, optional): MUAPs.
        properties_dict (dict, optional): Dictionary of motor unit properties.
        
    Returns:
        dict: Paths to saved files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get subject ID for filename prefix
    subject_id = metadata["simulation_info"].get("subject_id", "")
    subject_prefix = f"{subject_id}_" if subject_id else ""
    muscle = metadata["simulation_info"].get("target_muscle", "")
    
    # Prepare output paths with subject ID prefix
    paths = {
        'emg': os.path.join(output_dir, f'{subject_prefix}{muscle}_emg.npy'),
        'spikes': os.path.join(output_dir, f'{subject_prefix}{muscle}_spikes.npy'),
        'effort': os.path.join(output_dir, f'{subject_prefix}{muscle}_effort_profile.npy'),
        'config': os.path.join(output_dir, f'{subject_prefix}{muscle}_config_used.json'),
        'metadata': os.path.join(output_dir, f'{subject_prefix}{muscle}_metadata.json')
    }
    
    if angle_profile is not None:
        paths['angle'] = os.path.join(output_dir, f'{subject_prefix}{muscle}_angle_profile.npy')
    
    if muaps is not None:
        paths['muaps'] = os.path.join(output_dir, f'{subject_prefix}{muscle}_muaps.npy')
    
    if properties_dict is not None:
        paths['properties'] = os.path.join(output_dir, f'{subject_prefix}{muscle}_mn_properties.csv')
        save_motor_unit_properties_to_csv(properties_dict, paths['properties'])
    
    # Save all data
    np.save(paths['emg'], emg)
    np.save(paths['spikes'], spikes)
    np.save(paths['effort'], ext)
    
    if angle_profile is not None:
        np.save(paths['angle'], angle_profile)
    
    if muaps is not None:
        np.save(paths['muaps'], muaps)
    
    # Save configuration and metadata
    with open(paths['config'], 'w') as f:
        json.dump(cfg, f, indent=2)
    
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary of saved files
    print(f"Data saved to:")
    for key, path in paths.items():
        print(f"- {key}: {path}")
    
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EMG signals from movements')
    parser.add_argument('config_path', type=str, help='Path to input configuration JSON file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory (optional)', default=None)
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config_path)
    
    # Set default values from config or use defaults
    model_pth = cfg.get('PathToBioMimeWeights', './ckp/model_linear.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    morph = cfg.get('MorphMUAPS', False)
    muap_file = cfg.get('PathToMUAPFile', './ckp/muap_examples.pkl')
    
    # Subject configuration
    subject_cfg = cfg.SubjectConfiguration
    fibre_density = subject_cfg.FibreDensity
    subject_seed = subject_cfg.SubjectSeed
    subject_id = subject_cfg.get('SubjectID', f"subject_{subject_seed}")
    
    # Create muscle-to-motor-unit mapping from the config
    muscle_mu_map = {}
    if hasattr(subject_cfg, 'MuscleLabels') and hasattr(subject_cfg, 'MuscleMotorUnitCounts'):
        if len(subject_cfg.MuscleLabels) == len(subject_cfg.MuscleMotorUnitCounts):
            for muscle, count in zip(subject_cfg.MuscleLabels, subject_cfg.MuscleMotorUnitCounts):
                muscle_mu_map[muscle] = count
            print(f"Loaded motor unit counts for {len(muscle_mu_map)} muscles")
        else:
            print("Warning: MuscleLabels and MuscleMotorUnitCounts have different lengths. Using defaults.")
    
    # Set random seed for reproducibility
    print(f"Using subject seed: {subject_seed}")
    np.random.seed(subject_seed)
    torch.manual_seed(subject_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(subject_seed)
    
    # Movement configuration
    movement_cfg = cfg.MovementConfiguration
    ms_label = movement_cfg.TargetMuscle
    movement_duration = movement_cfg.MovementProfileParameters.MovementDuration
    
    # Recording configuration
    recording_cfg = cfg.RecordingConfiguration
    fs = recording_cfg.SamplingFrequency
    electrode_cfg = recording_cfg.ElectrodeConfiguration
    filter_cfg = recording_cfg.FilterProperties
    noise_seed = recording_cfg.NoiseSeed
    noise_level_db = recording_cfg.NoiseLeveldb
    
    # Configure electrode array
    n_electrodes = electrode_cfg.NElectrodes
    n_rows = electrode_cfg.NRows
    n_cols = electrode_cfg.NCols
    
    # Get desired number of columns (if specified)
    desired_cols = electrode_cfg.get('DesiredNCols', n_cols)
    
    # Determine the number of motor units to use
    if ms_label in muscle_mu_map:
        # Use muscle-specific MU count from mapping
        num_mus = muscle_mu_map[ms_label]
        print(f"Using configured motor unit count for {ms_label}: {num_mus}")
    else:
        # Fallback to default or configuration value
        num_mus = NUM_MUS.get(ms_label, 100)
        print(f"Using default motor unit count for {ms_label}: {num_mus}")
    
    # Setup MUAP cache with subject-specific path
    use_cache = args.cache_dir is not None
    if use_cache:
        muap_cache_dir = args.cache_dir
        os.makedirs(muap_cache_dir, exist_ok=True)

        muap_cache_file = os.path.join(muap_cache_dir, f"{subject_id}_{ms_label}_{movement_cfg.MovementDOF}_muaps.npy")
        muap_meta_file = os.path.join(muap_cache_dir, f"{subject_id}_{ms_label}_{movement_cfg.MovementDOF}_metadata.json")
        properties_csv_file = os.path.join(muap_cache_dir, f"{subject_id}_{ms_label}_{movement_cfg.MovementDOF}_mn_properties.csv")

        # Required metadata for cache compatibility check
        required_metadata = {
            'muscle': ms_label,
            'fs': fs,
            'electrode_rows': n_rows,
            'electrode_cols': n_cols,
            'subject_id': subject_id
        }

        # Try to load cached MUAPs
        muaps, use_cached_muaps, cached_num_mus, properties_dict = load_cached_muaps(
            muap_cache_file, 
            muap_meta_file, 
            required_metadata
        )
    else:
        use_cached_muaps = False

    # If no compatible cache, generate new MUAPs
    if not use_cached_muaps:
        # Build movement profile for MUAP generation
        fs_mov = 2  # Temporal frequency for movement simulation
        poses, durations, total_duration, steps = build_movement_profile(movement_cfg)
        
        # Generate MUAPs with subject-specific seed and MU count
        muaps, num_mus, properties_dict = generate_muaps(
            model_pth, ms_label, movement_cfg, fs_mov, poses, durations, steps,
            device, morph, muap_file, fibre_density, fs, filter_cfg, n_rows, n_cols,
            num_mus, subject_seed
        )
        
        # Cache the MUAPs for future use if cache_dir is provided
        if use_cache:
            muap_cache_metadata = {
                "subject_id": subject_id,
                "muscle": ms_label,
                "movement_dof": movement_cfg.MovementDOF,
                "fs": fs,
                "num_mus": num_mus,
                "device": device,
                "electrode_rows": n_rows,
                "electrode_cols": n_cols,
                "fiber_density": fibre_density,
                "subject_seed": subject_seed,
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            cache_muaps(muaps, muap_cache_file, muap_cache_metadata, muap_meta_file, properties_dict, properties_csv_file)
    
    else:
        # If using cached MUAPs, make sure we use the cached number of units
        num_mus = cached_num_mus
    
    # Generate angle profile and angle labels for MUAPs
    profile_params = movement_cfg.MovementProfileParameters
    angle_profile, muap_angle_labels = generate_angle_profile(
        fs, 
        movement_duration, 
        profile_params, 
        movement_cfg.MovementDOF,
        muaps.shape[1]
    )
    
    # Create effort profile
    effort_profile = create_effort_profile(fs, movement_duration, profile_params)
    
    # Initialize motoneuron pool for spike generation
    mn_pool = MotoneuronPool(num_mus, ms_label, **mn_default_settings)
    
    # Generate spike trains
    _, spikes, fr, ipis = generate_spike_trains(mn_pool, effort_profile, fs)
    
    # Generate EMG signal using the angle profile
    emg = generate_emg_signal(
        muaps, 
        spikes, 
        len(effort_profile), 
        muap_angle_labels, 
        angle_profile, 
        noise_level_db, 
        noise_seed
    )
    
    # Select optimal electrode columns if desired_cols is different from n_cols
    original_emg = emg.copy()
    selected_cols = None
    center_column = None
    
    if desired_cols < n_cols:
        print(f"Selecting {desired_cols} optimal columns from {n_cols}...")
        emg, selected_cols, center_column = select_optimal_electrode_columns(
            emg, 
            n_rows, 
            n_cols, 
            desired_cols
        )
        # Update n_cols to reflect the selected number
        n_cols = desired_cols
    
    print(emg.shape)
    # Create metadata
    steps_value = steps if 'steps' in locals() else "unknown"
    metadata = {
        "simulation_info": {
            "num_motor_units": num_mus,
            "movement_steps": steps_value,
            "target_muscle": ms_label,
            "fs": fs,
            "electrode_array": {
                "rows": n_rows,
                "columns": n_cols,
                "total_electrodes": n_rows * n_cols,
                "original_columns": electrode_cfg.NCols,
                "selected_columns": selected_cols,
                "center_column": center_column
            },
            "noise_level_db": noise_level_db,
            "movement_duration": movement_duration,
            "movement_dof": movement_cfg.MovementDOF,
            "movement_type": movement_cfg.MovementType,
            "effort_profile": profile_params.EffortProfile if hasattr(profile_params, 'EffortProfile') else "Default",
            "effort_level": profile_params.EffortLevel,
            "angle_profile": profile_params.AngleProfile if hasattr(profile_params, 'AngleProfile') else "Constant",
            "target_angle": getattr(profile_params, 'TargetAngle', 0) if hasattr(profile_params, 'AngleProfile') and profile_params.AngleProfile == "Constant" else None,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "muap_cache_used": use_cached_muaps,
            "muap_cache_path": muap_cache_file if use_cached_muaps else None,
            "subject_id": subject_id,
            "subject_seed": subject_seed,
            "muscle_motor_unit_mapping": muscle_mu_map
        }
    }

    
    # Save only new MUAPs if we generated them
    save_muaps = None if use_cached_muaps else muaps
    # Save all outputs
    save_outputs(
        args.output_dir, 
        emg, 
        spikes, 
        effort_profile, 
        cfg, 
        metadata, 
        angle_profile, 
        save_muaps,
        properties_dict
    )
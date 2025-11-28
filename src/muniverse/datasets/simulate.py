import json
import os
import shutil
import subprocess
import time
import numpy as np
import tempfile

from ..utils.logging import SimulationLogger
from .movement import generate_effort_profile, generate_angle_profile


def validate_config(config_content, verbose=False):
    """
    Validate the configuration dictionary to ensure all required parameters are present and valid.
    
    Args:
        config_content (dict): Configuration dictionary to validate
        verbose (bool, optional): If True, print validation success message. Defaults to False.
        
    Raises:
        ValueError: If configuration is invalid with specific error message
    """
    # Check required top-level sections
    required_sections = ["SubjectConfiguration", "MovementConfiguration", "RecordingConfiguration"]
    for section in required_sections:
        if section not in config_content:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate SubjectConfiguration
    subject_cfg = config_content["SubjectConfiguration"]
    required_subject_params = ["SubjectSeed", "FibreDensity", "MuscleLabels", "MuscleMotorUnitCounts"]
    for param in required_subject_params:
        if param not in subject_cfg:
            raise ValueError(f"Missing required parameter in SubjectConfiguration: {param}")
    
    # Validate subject seed
    if not isinstance(subject_cfg["SubjectSeed"], int) or subject_cfg["SubjectSeed"] < 0:
        raise ValueError("SubjectSeed must be a non-negative integer")
    
    # Validate fibre density
    if not isinstance(subject_cfg["FibreDensity"], (int, float)) or not (100 <= subject_cfg["FibreDensity"] <= 300):
        raise ValueError("FibreDensity must be a number between 100 and 300")
    
    # Validate muscle labels and counts match
    muscle_labels = subject_cfg["MuscleLabels"]
    muscle_counts = subject_cfg["MuscleMotorUnitCounts"]
    if len(muscle_labels) != len(muscle_counts):
        raise ValueError("MuscleLabels and MuscleMotorUnitCounts must have the same length")
    
    # Validate MovementConfiguration
    movement_cfg = config_content["MovementConfiguration"]
    required_movement_params = ["TargetMuscle", "MovementDOF", "MovementProfileParameters"]
    for param in required_movement_params:
        if param not in movement_cfg:
            raise ValueError(f"Missing required parameter in MovementConfiguration: {param}")
    
    # Validate target muscle is in the muscle labels
    # TODO: This doesn't work for hybrid pipeline.
    if movement_cfg["TargetMuscle"] not in muscle_labels:
        raise ValueError(f"TargetMuscle '{movement_cfg['TargetMuscle']}' not found in MuscleLabels")
    
    # Validate movement DOF
    valid_dofs = ["Flexion-Extension", "Radial-Ulnar-Deviation"]
    if movement_cfg["MovementDOF"] not in valid_dofs:
        raise ValueError(f"MovementDOF must be one of: {valid_dofs}")
    
    # Validate movement profile parameters
    profile_params = movement_cfg["MovementProfileParameters"]
    required_profile_params = ["MovementDuration", "TargetEffort"]
    for param in required_profile_params:
        if param not in profile_params:
            raise ValueError(f"Missing required parameter in MovementProfileParameters: {param}")
    
    # Validate movement duration
    if not isinstance(profile_params["MovementDuration"], (int, float)) or profile_params["MovementDuration"] <= 0:
        raise ValueError("MovementDuration must be a positive number")
    
    # Validate target effort
    if not isinstance(profile_params["TargetEffort"], (int, float)) or not (1 <= profile_params["TargetEffort"] <= 100):
        raise ValueError("TargetEffort must be a number between 1 and 100")
    
    # Validate RecordingConfiguration
    recording_cfg = config_content["RecordingConfiguration"]
    required_recording_params = ["SamplingFrequency", "ElectrodeConfiguration", "FilterProperties"]
    for param in required_recording_params:
        if param not in recording_cfg:
            raise ValueError(f"Missing required parameter in RecordingConfiguration: {param}")
    
    # Validate sampling frequency
    if not isinstance(recording_cfg["SamplingFrequency"], (int, float)) or recording_cfg["SamplingFrequency"] <= 0:
        raise ValueError("SamplingFrequency must be a positive number")
    
    # TODO: Validate filter properties
    # TODO: Validate electrode configuration
    
    if verbose:
        print("[INFO] Configuration validation passed")


def generate_neuromotion_recording(
    config, effort_profile, angle_profile, engine, container, logger=None, verbose=False
):
    """
    Generate a neuromotion recording using provided configuration and movement profiles.

    Args:
        config (dict): Configuration dictionary containing movement and recording parameters.
        effort_profile (np.ndarray): Effort profile array.
        angle_profile (np.ndarray): Angle profile array.
        engine (str): Container engine to use ("docker" or "singularity").
        container (str):
            For Docker: Name of the container image (e.g., "pranavm19/muniverse:neuromotion")
            For Singularity: Full path to the container file
        logger (SimulationLogger, optional): Logger instance. If None, a new logger will be created.
        verbose (bool, optional): If True, enable verbose logging. Defaults to False.

    Returns:
        dict: Dictionary containing simulation outputs:
            - 'emg': Generated EMG signal
            - 'spikes': Spike trains for each motor unit
            - 'firing_rates': Firing rates for each motor unit
            - 'effort_profile': Effort profile used
            - 'angle_profile': Angle profile used
            - 'muaps': MUAPs library
            - 'muap_angle_labels': Angle labels for MUAPs
            - 'properties': Motor unit properties
            - 'config': Configuration dictionary
    """
    # Initialize logger if not provided
    if logger is None:
        logger = SimulationLogger()
        logger.set_config(config)

    run_dir = tempfile.mkdtemp()

    # Package data for container - separate config and data files
    # Save config as JSON
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Save data arrays as NPZ file
    input_data_path = os.path.join(run_dir, "input_data.npz")
    np.savez(input_data_path, effort_profile=effort_profile, angle_profile=angle_profile)

    # Get the absolute path to the new script and shell script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "_run_neuromotion_new.py")
    run_script_path = os.path.join(current_dir, "_generate_recording_new.sh")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")

    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"Shell script not found at {run_script_path}")

    # Build command -
    # Pass the run directory containing config.json and input_data.npz
    cmd = [run_script_path, engine, container, script_path, run_dir]

    # Execute the shell script using subprocess
    try:
        subprocess.run(cmd, check=True, cwd=current_dir)
        print(f"[INFO] Data generated successfully at {run_dir}")
        if logger is not None:
            logger.set_return_code("run.sh", 0)
            logger.finalize(run_dir, engine, container)
        results = dict(np.load(os.path.join(run_dir, "emg_data.npz")))
        os.removedirs(run_dir)
        return results
    
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data generation failed: {e}")
        if verbose:
            print(f"[ERROR] Command output: {e.output}")
            print(f"[ERROR] Command stderr: {e.stderr}")
        if logger is not None:
            logger.set_return_code("run.sh", e.returncode)
            logger.finalize(run_dir, engine, container)
        os.removedirs(run_dir)
        raise


def generate_hybrid_recording(
    config, effort_profile, angle_profile, muaps, muap_angle_labels, run_dir, engine, container, logger=None, verbose=False
):
    """
    Generate a hybrid recording using provided configuration, movement profiles, and MUAPs.

    Args:
        config (dict): Configuration dictionary containing movement and recording parameters.
        effort_profile (np.ndarray): Effort profile array.
        angle_profile (np.ndarray): Angle profile array.
        muaps (np.ndarray): MUAPs array. Can be provided in two formats:
            - Shape (n_motor_units, n_angle_labels, n_electrodes, n_timepoints): Will be reshaped to grid format
            - Shape (n_motor_units, n_angle_labels, ch_rows, ch_cols, n_timepoints): Used directly
        muap_angle_labels (np.ndarray): Angle labels array of length n_angle_labels describing what angle each MUAP corresponds to.
        engine (str): Container engine to use ("docker" or "singularity").
        container (str):
            For Docker: Name of the container image (e.g., "pranavm19/muniverse:neuromotion")
            For Singularity: Full path to the container file
        logger (SimulationLogger, optional): Logger instance.
        verbose (bool, optional): If True, enable verbose logging. Defaults to False.

    Returns:
        dict: Dictionary containing simulation outputs:
            - 'emg': Generated EMG signal
            - 'spikes': Spike trains for each motor unit
            - 'firing_rates': Firing rates for each motor unit
            - 'effort_profile': Effort profile used
            - 'angle_profile': Angle profile used
            - 'muaps': MUAPs library
            - 'muap_angle_labels': Angle labels for MUAPs
            - 'properties': Motor unit properties
            - 'config': Configuration dictionary
    """
    # Validate required parameters
    if muap_angle_labels is None:
        raise ValueError("'muap_angle_labels' is a required parameter for hybrid recording")

    # Initialize logger if not provided
    if logger is not None:
        logger.set_config(config)

    run_dir = tempfile.mkdtemp()

    # Package data for container - separate config and data files
    # Save config as JSON
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Save data arrays as NPZ file - ensure arrays are numpy arrays
    effort_profile_array = np.asarray(effort_profile)
    angle_profile_array = np.asarray(angle_profile)
    muaps_array = np.asarray(muaps)
    muap_angle_labels = np.asarray(muap_angle_labels)
    
    # Validate muaps shape and reshape if needed
    # Container expects shape (n_units, n_angle_labels, ch_rows, ch_cols, n_timepoints)
    if len(muaps_array.shape) == 4:
        # Shape is (n_motor_units, n_angle_labels, n_electrodes, n_timepoints)
        # Need to reshape to (n_motor_units, n_angle_labels, ch_rows, ch_cols, n_timepoints)
        n_motor_units, n_angle_labels, n_electrodes, n_timepoints = muaps_array.shape
        
        # Get electrode grid dimensions from config
        electrode_cfg = config.get("RecordingConfiguration", {}).get("ElectrodeConfiguration", {})
        ch_rows = electrode_cfg.get("NRows")
        ch_cols = electrode_cfg.get("NCols")
        
        if ch_rows is None or ch_cols is None:
            # Try to infer from NElectrodes
            n_electrodes_config = electrode_cfg.get("NElectrodes")
            if n_electrodes_config and n_electrodes_config == n_electrodes:
                # Try to infer grid shape
                ch_rows = int(np.sqrt(n_electrodes))
                ch_cols = n_electrodes // ch_rows
                if ch_rows * ch_cols != n_electrodes:
                    raise ValueError(
                        f"Cannot infer grid shape from n_electrodes={n_electrodes}. "
                        "Please provide NRows and NCols in ElectrodeConfiguration."
                    )
            else:
                raise ValueError(
                    "Cannot reshape muaps: NRows and NCols must be specified in ElectrodeConfiguration, "
                    f"or NElectrodes ({n_electrodes_config}) must match n_electrodes ({n_electrodes})"
                )
        
        if ch_rows * ch_cols != n_electrodes:
            raise ValueError(
                f"Electrode grid dimensions ({ch_rows}x{ch_cols}={ch_rows*ch_cols}) "
                f"do not match muaps electrode dimension ({n_electrodes})"
            )
        
        # Reshape from (n_motor_units, n_angle_labels, n_electrodes, n_timepoints)
        # to (n_motor_units, n_angle_labels, ch_rows, ch_cols, n_timepoints)
        muaps_array = muaps_array.reshape(n_motor_units, n_angle_labels, ch_rows, ch_cols, n_timepoints)
        
    elif len(muaps_array.shape) == 5:
        # Already in correct format (n_motor_units, n_angle_labels, ch_rows, ch_cols, n_timepoints)
        pass
    else:
        raise ValueError(
            f"muaps must have 4 or 5 dimensions, got shape {muaps_array.shape}. "
            "Expected: (n_motor_units, n_angle_labels, n_electrodes, n_timepoints) or "
            "(n_motor_units, n_angle_labels, ch_rows, ch_cols, n_timepoints)"
        )
    
    # Validate muap_angle_labels length matches muaps
    if muaps_array.shape[1] != len(muap_angle_labels):
        raise ValueError(
            f"muaps second dimension ({muaps_array.shape[1]}) must match muap_angle_labels length ({len(muap_angle_labels)})"
        )
    
    # For hybrid, we need to include muaps and muap_angle_labels in input_data.npz
    input_data_path = os.path.join(run_dir, "input_data.npz")
    np.savez(input_data_path, 
             effort_profile=effort_profile_array, 
             angle_profile=angle_profile_array,
             muaps=muaps_array,
             muap_angle_labels=muap_angle_labels)

    # Get the absolute path to the script and shell script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "_run_neuromotion_new.py")
    run_script_path = os.path.join(current_dir, "_generate_recording_new.sh")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")

    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"Shell script not found at {run_script_path}")

    # Build command - same as neuromotion since _run_neuromotion_new.py handles both
    cmd = [run_script_path, engine, container, script_path, run_dir]

    # Execute the shell script using subprocess
    try:
        subprocess.run(cmd, check=True, cwd=current_dir)
        print(f"[INFO] Hybrid recording generated successfully at {run_dir}")
        if logger is not None:
            logger.set_return_code("run_hybrid", 0)
            logger.finalize(run_dir, engine, container)
        results = dict(np.load(os.path.join(run_dir, "emg_data.npz")))
        os.removedirs(run_dir)
        return results
    
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Hybrid recording generation failed: {e}")
        if verbose:
            print(f"[ERROR] Command output: {e.output}")
            print(f"[ERROR] Command stderr: {e.stderr}")
        if logger is not None:
            logger.set_return_code("run_hybrid", e.returncode)
            logger.finalize(run_dir, engine, container)
        os.removedirs(run_dir)
        raise


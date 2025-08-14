import json
import os
import shutil
import subprocess
import time
import numpy as np

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
    required_profile_params = ["MovementDuration", "EffortLevel"]
    for param in required_profile_params:
        if param not in profile_params:
            raise ValueError(f"Missing required parameter in MovementProfileParameters: {param}")
    
    # Validate movement duration
    if not isinstance(profile_params["MovementDuration"], (int, float)) or profile_params["MovementDuration"] <= 0:
        raise ValueError("MovementDuration must be a positive number")
    
    # Validate effort level
    if not isinstance(profile_params["EffortLevel"], (int, float)) or not (1 <= profile_params["EffortLevel"] <= 100):
        raise ValueError("EffortLevel must be a number between 1 and 100")
    
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
    input_config, output_dir, engine, container, cache_dir=None, verbose=False
):
    """
    Generate a neuromotion recording using the specified configuration file.

    Args:
        input_config (str): Path to the JSON configuration file containing movement and recording parameters.
        output_dir (str): Path to the output directory where the generated data will be saved.
        engine (str): Container engine to use
        container (str):
            For Docker: Name of the container image (e.g., "muniverse-test:neuromotion")
            For Singularity: Full path to the container file
        cache_dir (str, optional): Path to cache directory. If None, no caching is used.
        verbose (bool, optional): If True, enable verbose logging. Defaults to False.
    """
    # Initialize logger
    logger = SimulationLogger()

    # Load and log configuration
    with open(input_config, "r") as f:
        config_content = json.load(f)

    # Validate configuration file
    validate_config(config_content, verbose)

    logger.set_config(config_content)

    # Convert paths to absolute paths
    input_config = os.path.abspath(input_config)
    output_dir = os.path.abspath(output_dir)

    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Generate movement profiles from config
    effort_profile = generate_effort_profile(config_content)
    angle_profile = generate_angle_profile(config_content)

    # Package data for container - separate config and data files
    # Save config as JSON
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_content, f)
    
    # Save data arrays as NPZ file - ensure arrays are numpy arrays
    effort_profile_array = np.asarray(effort_profile)
    angle_profile_array = np.asarray(angle_profile)
    
    input_data_path = os.path.join(run_dir, "input_data.npz")
    np.savez(input_data_path, effort_profile=effort_profile_array, angle_profile=angle_profile_array)

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
        subprocess.run(
            cmd,
            check=True,
            cwd=current_dir,
            # capture_output=True,
            # text=True
        )
        print(f"[INFO] Data generated successfully at {run_dir}")
        logger.set_return_code("run.sh", 0)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data generation failed: {e}")
        if verbose:
            print(f"[ERROR] Command output: {e.output}")
            print(f"[ERROR] Command stderr: {e.stderr}")
        logger.set_return_code("run.sh", e.returncode)
        raise

    # TODO: Add post-processing steps and finalize log and saving

    # Finalize and save the log
    log_path = logger.finalize(run_dir, engine, container)
    if verbose:
        print(f"Run log saved to: {log_path}")

    return run_dir


def generate_hybrid_recording(
    input_config, muaps_file, output_dir, engine, container, cache_dir
):
    """
    Generate a hybrid recording using only the spike generation from neuromotion.
    User needs to provide a MUAPs file.

    Args:
        input_config (str): Path to the JSON configuration file
        muaps_file (str): Path to the MUAPs file
        output_dir (str): Path to the output directory
        engine (str): Container engine to use (docker or singularity)
        container (str): Container image or SIF file path
        cache_dir (str): Path to the cache directory containing MUAPs files

    Returns:
        str: Path to the output directory
    """
    # Initialize logger
    logger = SimulationLogger()

    # Load and log configuration
    with open(input_config, "r") as f:
        config_content = json.load(f)
    logger.set_config(config_content)

    # Convert paths to absolute paths
    input_config = os.path.abspath(input_config)
    output_dir = os.path.abspath(output_dir)
    cache_dir = os.path.abspath(cache_dir)

    # Define the MUAPs file path
    muaps_file = os.path.join(cache_dir, "hybrid_TA_muaps.npz")

    if not os.path.exists(muaps_file):
        raise FileNotFoundError(f"MUAPs file not found at {muaps_file}")

    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Get the absolute path to the script and shell script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "_run_hybrid.py")
    run_script_path = os.path.join(current_dir, "_generate_records_hybrid.sh")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")

    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"Shell script not found at {run_script_path}")

    # Build command with the shell script
    cmd = [
        run_script_path,
        engine,
        container,
        script_path,
        input_config,
        run_dir,
        muaps_file,
    ]

    # Execute the shell script using subprocess
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=current_dir,
        )
        print(f"[INFO] Hybrid tibialis data generated successfully at {run_dir}")
        logger.set_return_code("run_hybrid", 0)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Hybrid tibialis data generation failed: {e}")
        logger.set_return_code("run_hybrid", e.returncode)
        raise

    # Log output files
    for root, _, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))

    # Finalize and save the log
    log_path = logger.finalize(run_dir, engine, container)
    print(f"Run log saved to: {log_path}")

    return run_dir

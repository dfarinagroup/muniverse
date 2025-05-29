import subprocess
import os
import json
import time
import shutil
from ..utils.logging import SimulationLogger


def generate_neuromotion_recording(input_config, output_dir, engine, container, cache_dir=None):
    """
    Generate a neuromotion recording using the specified configuration file.
    TODO: Add verbose flag to control logging

    Args:
        input_config (str): Path to the JSON configuration file containing movement and recording parameters.
        output_dir (str): Path to the output directory where the generated data will be saved.
        engine (str): Container engine to use
        container (str): 
            For Docker: Name of the container image (e.g., "muniverse-test:neuromotion")
            For Singularity: Full path to the container file
        cache_dir (str, optional): Path to cache directory. If None, no caching is used.
    """
    # Initialize logger
    logger = SimulationLogger()
    
    # Load and log configuration
    with open(input_config, 'r') as f:
        config_content = json.load(f)
    logger.set_config(config_content)
    
    # Convert paths to absolute paths
    input_config = os.path.abspath(input_config)
    output_dir = os.path.abspath(output_dir)
    
    # Get the absolute path to the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "_run_neuromotion.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")
    
    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Get the correct path to run.sh
    run_script_path = os.path.join(current_dir, "_generate_recording.sh")
    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"run.sh not found at {run_script_path}")

    # Build command with optional cache directory
    cmd = [run_script_path, engine, container, script_path, input_config, run_dir]
    if cache_dir is not None:
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cmd.append(cache_dir)
    
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
        print(f"[ERROR] Command output: {e.output}")
        print(f"[ERROR] Command stderr: {e.stderr}")
        logger.set_return_code("run.sh", e.returncode)
        raise
    
    # Load runtime metadata from container
    metadata_files = [f for f in os.listdir(run_dir) if f.endswith('_metadata.json')]
    if metadata_files:
        metadata_path = os.path.join(run_dir, metadata_files[0])
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Extract center column from metadata
        if 'simulation_info' in metadata and 'electrode_array' in metadata['simulation_info']:
            center_column = metadata['simulation_info']['electrode_array'].get('center_column', 'N/A')
            print(f"[INFO] Found center column: {center_column}")
            logger.log_data['OutputData']['Metadata']['CenterColumn'] = center_column
    
    # Log output files
    for root, _, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))
    
    # Finalize and save the log
    log_path = logger.finalize(run_dir, engine, container)
    print(f"Run log saved to: {log_path}")
    
    return run_dir


def generate_hybrid_tibialis_recording(input_config, output_dir, engine, container, cache_dir):
    """
    Generate a hybrid tibialis recording using the _run_hybrid.py script.
    
    Args:
        input_config (str): Path to the JSON configuration file
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
    with open(input_config, 'r') as f:
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
        muaps_file
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
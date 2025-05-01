import subprocess
import os
import json
import time
import shutil
from ..utils.logging import RunLogger


def generate_neuromotion_recording(input_config, output_dir, engine="singularity", container_name="environment/muniverse-test_neuromotion.sif", cache_dir=None):
    """
    Generate a neuromotion recording using the specified configuration file.

    Args:
        input_config (str): Path to the JSON configuration file containing movement and recording parameters.
        output_dir (str): Path to the output directory where the generated data will be saved.
        engine (str, optional): Container engine to use ("docker" or "singularity"). Defaults to "singularity".
        container_name (str, optional): 
            For Docker: Name of the container image (e.g., "muniverse-test:neuromotion")
            For Singularity: Path to the container file (e.g., "environment/muniverse-test_neuromotion.sif")
            Defaults to "environment/muniverse-test_neuromotion.sif".
        cache_dir (str, optional): Path to cache directory. If None, no caching is used.
    """
    # Initialize logger
    logger = RunLogger()
    
    # Load and log configuration
    with open(input_config, 'r') as f:
        config_content = json.load(f)
    logger.set_config(config_content)
    
    # Get container info
    if engine == "docker":
        try:
            inspect_output = subprocess.check_output([engine, "inspect", container_name]).decode()
            inspect_data = json.loads(inspect_output)[0]
            image_info = {
                "name": inspect_data["RepoTags"][0] if inspect_data["RepoTags"] else "unknown",
                "id": inspect_data["Id"],
                "created": inspect_data["Created"]
            }
        except Exception as e:
            print(f"Warning: Could not get container info: {e}")
            image_info = {"name": "unknown", "id": "unknown", "created": "unknown"}
    else:
        image_info = {"name": "unknown", "id": "unknown", "created": "unknown"}

    # Set container info
    logger.set_container_info(
        engine=engine,
        engine_version=subprocess.check_output([engine, "--version"]).decode().strip(),
        image=image_info["name"],
        image_id=image_info["id"]
    )
    
    # Convert paths to absolute paths
    input_config = os.path.abspath(input_config)
    output_dir = os.path.abspath(output_dir)
    
    # Get the absolute path to the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "run_neuromotion_extended.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")
    
    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Get the correct path to run.sh
    run_script_path = os.path.join(current_dir, "run.sh")
    if not os.path.exists(run_script_path):
        raise FileNotFoundError(f"run.sh not found at {run_script_path}")
    
    # Handle container name based on engine type
    if engine.lower() == "singularity":
        container_name = os.path.abspath(container_name)
    
    # Build command with optional cache directory
    cmd = [run_script_path, engine, container_name, script_path, input_config, run_dir]
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
            capture_output=True,
            text=True
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
    metadata_path = os.path.join(run_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Log output files
    for root, _, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))
    
    # Finalize and save the log
    log_path = logger.finalize(run_dir)
    print(f"Run log saved to: {log_path}")
    
    return run_dir

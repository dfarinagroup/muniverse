import subprocess
import os
import json
from easydict import EasyDict as edict
from src.utils.logging import RunLogger


def generate_dataset(input_config, output_dir, engine="singularity", container_name="environment/muniverse-test_neuromotion.sif"):
    """
    Generate a dataset using the specified configuration file.

    Args:
        input_config (str): Path to the JSON configuration file containing movement and recording parameters.
        output_dir (str): Path to the output directory where the generated data will be saved.
        engine (str, optional): Container engine to use ("docker" or "singularity"). Defaults to "singularity".
        container_name (str, optional): 
            For Docker: Name of the container image (e.g., "muniverse-test:neuromotion")
            For Singularity: Path to the container file (e.g., "environment/muniverse-test_neuromotion.sif")
            Defaults to "environment/muniverse-test_neuromotion.sif".

    Returns:
        str: Absolute path to the output directory containing the generated dataset.
    """
    # Initialize logger
    logger = RunLogger()
    
    # Load and log configuration
    with open(input_config, 'r') as f:
        config_content = json.load(f)
    logger.set_config(config_content)
    
    # Set container information
    logger.set_container_info(
        engine=engine,
        engine_version=subprocess.check_output([engine, "--version"]).decode().strip(),
        image=container_name,
        image_id=subprocess.check_output([engine, "inspect", container_name]).decode().strip() if engine == "docker" else "unknown"
    )
    
    # Convert paths to absolute paths
    input_config = os.path.abspath(input_config)
    output_dir = os.path.abspath(output_dir)
    script_path = os.path.abspath("src/data_generation/run_neuromotion_extended.py")
    
    # Get the correct path to run.sh
    run_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.sh")
    
    # Handle container name based on engine type
    if engine.lower() == "singularity":
        container_name = os.path.abspath(container_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Execute the shell script using subprocess
    try:
        subprocess.run(
            [run_script_path, engine, container_name, script_path, input_config, output_dir],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"[INFO] Data generated successfully at {output_dir}")
        logger.set_return_code("run.sh", 0)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data generation failed: {e}")
        logger.set_return_code("run.sh", e.returncode)
        raise
    
    # Load runtime metadata from container
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Log output files
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))
    
    # Finalize and save the log
    
    log_path = logger.finalize(output_dir)
    print(f"Run log saved to: {log_path}")
    
    return output_dir

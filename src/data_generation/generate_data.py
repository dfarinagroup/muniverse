import subprocess
import os
from easydict import EasyDict as edict


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
    # Convert paths to absolute paths
    input_config = os.path.abspath(input_config)
    output_dir = os.path.abspath(output_dir)
    script_path = os.path.abspath("src/data_generation/run_neuromotion_extended.py")
    
    # Get the correct path to run.sh
    run_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.sh")
    
    # Handle container name based on engine type
    if engine.lower() == "singularity":
        container_name = os.path.abspath(container_name)
    # For Docker, we keep the container name as is
    
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
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data generation failed: {e}")
        raise

    return output_dir

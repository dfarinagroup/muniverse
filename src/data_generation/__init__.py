"""
Data generation utilities for neuromotion.
""" 

from .generate_data import generate_neuromotion_recording
from src.utils.containers import pull_container, verify_container_engine

def init():
    """
    Initialize the datasets module.
    This includes verifying container engines and pulling container images if needed.
    If both Docker and Singularity are available, Singularity will be used by default.

    Returns:
        str: The selected container engine ("docker" or "singularity")
    """
    # Check availability of both engines
    docker_available = verify_container_engine("docker")
    singularity_available = verify_container_engine("singularity")
    
    # Select engine based on availability
    if singularity_available:
        engine = "singularity"
    elif docker_available:
        engine = "docker"
    else:
        raise RuntimeError("No container engine (Docker or Singularity) is available. Please install one first.")
    
    # Get container name (using default)
    container_name = "pranavm19/muniverse-test:neuromotion"
    
    # Pull container if needed
    pull_container(container_name, engine)
    print(f"[INFO] Datasets module initialized using {engine}.")
    
    return engine

def generate_recording(config):
    """
    Generate a neuromotion recording using the provided configuration.

    Args:
        config (dict): Configuration dictionary that should include:
            - input_config: Path to the JSON configuration file containing movement and recording parameters
            - output_dir: Path to the output directory where the generated data will be saved
            - engine (optional): Container engine to use ("docker" or "singularity"). Defaults to "singularity"
            - container_name (optional): 
                For Docker: Name of the container image (e.g., "muniverse-test:neuromotion")
                For Singularity: Path to the container file (e.g., "environment/muniverse-test_neuromotion.sif")
                Defaults to "environment/muniverse-test_neuromotion.sif"
            - cache_dir (optional): Path to cache directory. If None, no caching is used.

    Returns:
        str: The path to the generated dataset.
    """   
    # Extract required parameters
    input_config = config.get("input_config")
    output_dir = config.get("output_dir")
    
    if not input_config or not output_dir:
        raise ValueError("Both 'input_config' and 'output_dir' are required parameters")
    
    # Initialize containers first
    engine = config.get("engine", init())

    # Extract optional parameters with defaults
    if engine == "singularity":
        container_name = config.get("container_name", "environment/muniverse-test_neuromotion.sif")
    else:
        container_name = config.get("container_name", "pranavm19/muniverse-test:neuromotion")

    cache_dir = config.get("cache_dir", None)
    
    # Generate the dataset
    return generate_neuromotion_recording(input_config, output_dir, engine, container_name, cache_dir)

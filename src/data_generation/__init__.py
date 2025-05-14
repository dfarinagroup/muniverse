"""
Data generation utilities for neuromotion.
""" 

from .generate_data import generate_neuromotion_recording, generate_hybrid_tibialis_recording
from ..utils.containers import pull_container, verify_container_engine
import os
import json

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

def is_hybrid_tibialis_config(config_path):
    """
    Determine if a config file is for hybrid tibialis setup.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        bool: True if it's a hybrid tibialis config
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check if target muscle is Tibialis Anterior
        if config.get("MovementConfiguration", {}).get("TargetMuscle") == "Tibialis Anterior":
            return True
            
        # Check if config filename contains tibialis
        if "tibialis" in os.path.basename(config_path).lower():
            return True
            
        return False
    except:
        return False

def generate_recording(config):
    """
    Generate a neuromotion recording using the provided configuration.

    Args:
        config (dict): Configuration dictionary that should include:
            - input_config: Path to the JSON configuration file containing movement and recording parameters
            - output_dir: Path to the output directory where the generated data will be saved
            - engine: Container engine to use ("docker" or "singularity")
            - container: 
                For Docker: Name of the container image (e.g., "muniverse-test:neuromotion")
                For Singularity: Full path to the container file (e.g., "environment/muniverse-test_neuromotion.sif")
            - cache_dir: Path to cache directory. For hybrid tibialis, should contain hybrid_TA_muaps.npz.

    Returns:
        str: The path to the generated dataset.
    """   
    # Extract required parameters
    input_config = config.get("input_config")
    output_dir = config.get("output_dir")
    engine = config.get("engine")
    container = config.get("container")
    cache_dir = config.get("cache_dir")
    
    # Validate required parameters
    if not input_config or not output_dir:
        raise ValueError("Both 'input_config' and 'output_dir' are required parameters")
    
    if not engine or not container:
        raise ValueError("'engine' and 'container' are required parameters")
    
    if not cache_dir:
        raise ValueError("'cache_dir' is a required parameter")
    
    # Check if this is a hybrid tibialis config
    if is_hybrid_tibialis_config(input_config):
        # Generate using hybrid tibialis method
        return generate_hybrid_tibialis_recording(
            input_config, 
            output_dir,
            engine,
            container,
            cache_dir
        )
    else:
        # Generate using regular neuromotion method
        return generate_neuromotion_recording(
            input_config, 
            output_dir, 
            engine, 
            container, 
            cache_dir
        )
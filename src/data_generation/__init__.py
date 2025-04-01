"""
Data generation utilities for neuromotion.
""" 

import yaml
from .generate_data import generate_dataset
from src.utils.containers import pull_container, verify_container_engine

from easydict import EasyDict as edict

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def init(config=None):
    """
    Initialize the datasets module.
    This includes verifying container engines and pulling container images if needed.

    Args:
        config (dict, optional): Configuration dictionary that includes:
            - engine: "docker" or "singularity" (default: "docker")
            - container_name: Name of the container to pull (default: "pranavm19/muniverse-test:neuromotion")
    """
    if config is None:
        config = {}
    
    # Get engine from config or default to docker
    engine = config.get("engine", "docker")
    
    # Verify container engine is available
    if not verify_container_engine(engine):
        raise RuntimeError(f"Container engine '{engine}' is not available. Please install it first.")
    
    # Get container name from config or use default
    container_name = config.get("container_name", "pranavm19/muniverse-test:neuromotion")
    
    # Pull container if needed
    pull_container(container_name, engine)
    print("[INFO] Datasets module initialized.")

def generate(config):
    """
    Generate a dataset using the provided configuration.

    Args:
        config (dict): Configuration dictionary that should include:
            - engine: "docker" or "singularity"
            - sim_script: Path to the simulation script
            - output_dir: Where the generated data should be stored
            - container_name: Name of the container to use (optional)
            - (other parameters as needed)

    Returns:
        str: The path to the generated dataset.
    """
    # Initialize containers before generating data
    init(load_config(config))
    return generate_dataset(config)

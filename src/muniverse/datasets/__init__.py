"""
Data generation utilities for neuromotion.
"""

import json
import os
import time
import numpy as np

from ..utils.containers import pull_container, verify_container_engine
from ..utils.logging import SimulationLogger
from .simulate import (
    generate_hybrid_recording as _generate_hybrid_recording,
    generate_neuromotion_recording,
    validate_config,
)
from .movement import generate_effort_profile, generate_angle_profile


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
        raise RuntimeError(
            "No container engine (Docker or Singularity) is available. Please install one first."
        )

    # Get container name (using default)
    container_name = "pranavm19/muniverse:neuromotion"

    # Pull container if needed
    pull_container(container_name, engine)
    print(f"[INFO] Datasets module initialized using {engine}.")


def generate_synthetic_recording(
    input_config, output_dir, engine, container, cache_dir, verbose=False
):
    """
    Generate a synthetic neuromotion recording using the provided configuration file.
    This function handles all scaffolding: config loading, validation, profile generation,
    directory setup, and logging. It then calls the execution function.

    Args:
        input_config (str): Path to the JSON configuration file containing movement and recording parameters.
        engine (str): Container engine to use ("docker" or "singularity").
        container (str):
            For Docker: Name of the container image (e.g., "pranavm19/muniverse:neuromotion")
            For Singularity: Full path to the container file (e.g., "environment/muniverse_neuromotion.sif")
        output_dir (str, optional): Path to the output directory where the generated data will be saved. Defaults to None.
        verbose (bool, optional): If True, enable verbose logging. Defaults to False.

    Returns:
        dict: Dictionary containing simulation outputs (see generate_neuromotion_recording for details).
    """
    # Validate required parameters
    if not input_config or not output_dir:
        raise ValueError("Both 'input_config' and 'output_dir' are required parameters")

    if not engine or not container:
        raise ValueError("'engine' and 'container' are required parameters")

    # Initialize logger
    logger = SimulationLogger()

    # Load and validate configuration
    input_config = os.path.abspath(input_config)
    with open(input_config, "r") as f:
        config_content = json.load(f)

    # Validate configuration file
    validate_config(config_content, verbose)
    logger.set_config(config_content)

    # Generate movement profiles from config
    effort_profile = generate_effort_profile(config_content)
    angle_profile = generate_angle_profile(config_content)

    # Call neuromotion recording generation
    results = generate_neuromotion_recording(
        config=config_content,
        effort_profile=effort_profile,
        angle_profile=angle_profile,
        run_dir=run_dir,
        engine=engine,
        container=container,
        logger=logger,
        verbose=verbose,
    )

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        np.savez(os.path.join(output_dir, "emg_data.npz"), **results)
        print(f"[INFO] Data saved to {output_dir}")
    
    return results

def generate_hybrid_recording(
    input_config, output_dir, engine, container, cache_dir, muaps, muap_angle_labels, verbose=False
):
    """
    Generate a hybrid recording using provided MUAPs and angle labels.
    This function handles all scaffolding: config loading, validation, profile generation,
    directory setup, and logging. It then calls the execution function.

    Args:
        input_config (str): Path to the JSON configuration file containing movement and recording parameters.
        engine (str): Container engine to use ("docker" or "singularity").
        container (str):
            For Docker: Name of the container image (e.g., "pranavm19/muniverse:neuromotion")
            For Singularity: Full path to the container file (e.g., "environment/muniverse_neuromotion.sif")
        muaps (np.ndarray): MUAPs array with shape (n_motor_units, n_angle_labels, n_electrodes, n_timepoints).
            The electrode dimension can be a flattened grid (n_rows * n_cols) or a 2D grid that will be reshaped.
        muap_angle_labels (np.ndarray): Angle labels array of length n_angle_labels describing what angle each MUAP corresponds to.
        output_dir (str, optional): Path to the output directory where the generated data will be saved. Defaults to None.
        verbose (bool, optional): If True, enable verbose logging. Defaults to False.

    Returns:
        dict: Dictionary containing simulation outputs (see generate_hybrid_recording for details).
    """
    # Validate required parameters
    if not input_config or not output_dir:
        raise ValueError("Both 'input_config' and 'output_dir' are required parameters")

    if not engine or not container:
        raise ValueError("'engine' and 'container' are required parameters")

    if muaps is None:
        raise ValueError("'muaps' is a required parameter for hybrid recording")

    if muap_angle_labels is None:
        raise ValueError("'muap_angle_labels' is a required parameter for hybrid recording")

    # Validate muaps and muap_angle_labels shapes
    muaps = np.asarray(muaps)
    muap_angle_labels = np.asarray(muap_angle_labels)

    if len(muaps.shape) < 4:
        raise ValueError(f"muaps must have at least 4 dimensions, got shape {muaps.shape}")

    if muaps.shape[1] != len(muap_angle_labels):
        raise ValueError(
            f"muaps second dimension ({muaps.shape[1]}) must match muap_angle_labels length ({len(muap_angle_labels)})"
        )

    # Initialize logger
    logger = SimulationLogger()

    # Load and validate configuration
    input_config = os.path.abspath(input_config)
    with open(input_config, "r") as f:
        config_content = json.load(f)

    # Validate configuration file
    validate_config(config_content, verbose)
    logger.set_config(config_content)

    # Generate movement profiles from config
    effort_profile = generate_effort_profile(config_content)
    angle_profile = generate_angle_profile(config_content)

    # Call hybrid recording generation
    results = _generate_hybrid_recording(
        config=config_content,
        effort_profile=effort_profile,
        angle_profile=angle_profile,
        muaps=muaps,
        muap_angle_labels=muap_angle_labels,
        engine=engine,
        container=container,
        logger=logger,
        verbose=verbose,
    )

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        np.savez(os.path.join(output_dir, "emg_data.npz"), **results)
        print(f"[INFO] Data saved to {output_dir}")
    
    return results


def load_datasets():
    """
    Load BIDS-formatted datasets available through MUniverse

    Available datasets include:
    1. Avrillon et. al. 2023
    2. Caillet et. al. 2024
    3. Grison et. al. 2025
    4. MUniverse Neuromotion-Train set
    5. MUinverse Neuromotion-Test set
    6. MUniverse Hybrid-Tibialis set

    # TODO: Import from harvard dataverse

    """
    pass

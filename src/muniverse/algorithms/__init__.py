"""
Benchmark algorithms for decomposition.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Literal

import numpy as np
from pyedflib.highlevel import read_edf

from ..utils.containers import pull_container, verify_container_engine
from .decomposition import decompose_cbss, decompose_scd, decompose_ae


def init():
    """
    Initialize the algorithms module.
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
    container_name = "pranavm19/muniverse:scd"

    # Pull container if needed
    pull_container(container_name, engine)
    print(f"[INFO] Algorithms module initialized using {engine}.")

    return engine


def decompose_recording(
    data: Union[str, np.ndarray],
    fsamp: float, 
    method: Literal["cbss", "scd", "ae"] = "cbss",
    algorithm_config: Optional[Dict] = None,
    engine: Literal["local", "docker", "singularity"] = "local",
    container: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    API to decompose EMG recordings using the specified method.

    Args
    ----
        data : {str, np.ndarray} 
            Either a path to input data file (.npy or .edf) or a numpy array 
            of EMG data (n_channels, n_samples)

        fsamp : float
            Sampling rate in Hz    

        method: {"scd", "cbss", "ae"} , default "cbss"
            Decomposition method to use. Use "scd" for SwarmContrastiveDecomposition,
            "cbss" for FastIcaCBSS, or "ae" for AEDecoder.

        algorithm_config : dict 
            Dictionary containing the algorithm configuration. If None, the default 
            configs will be used

        engine : {"docker", "singularity", "local"}, default "local"
            Container engine to use. If "local", no container is used and code is 
            evaluated locally (only required for SCD method).

        container : str 
            Path to container image (only required for SCD method).

    Returns
    -------
        results : dict
            Dictonary containing
                - data (np.ndarray): Pre-processed data
                - spikes (pd.DataFrame): Table of motor unit spikes
                - sources (np.ndarray): Predicted sources
                - scores (dict): Source quality metrics
                - pre_process_metadata (dict): Metadata correspoding to
                pre processing steps (Optional)
                - post_process_metadata (dict): Metadata correspoding to
                post processing steps (Optional)

        log_data : dict
            Dictonary of processing metadata 
        
    Note
    ----
        For UpperBound decomposition, use decompose_upperbound(...) directly.
    """

    # Check input data
    if isinstance(data, str):
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"Input data file not found: {data_path}")
        if data_path.suffix not in [".npy", ".edf", ".bdf", ".edf+", ".bdf+"]:
            raise ValueError(
                f"Unsupported file format: {data_path.suffix}. Must be .npy or .edf"
            )

        # Load data into numpy array
        if data_path.suffix in [".edf", ".bdf", ".edf+", ".bdf+"]:            
            data = read_edf(data_path)
        else:  # .npy
            data = np.load(data_path)
    
    # Validate numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be either a file path (str) or numpy array")
    if data.ndim != 2:
        raise ValueError("EMG data must be a 2D array (channels x samples)")

    # Route to appropriate method
    if method == "scd":
        if (engine in ["docker", "singularity"] and container is None):
            raise ValueError(
                "Container path must be provided to run SCD in a container."
            )
                
        # Run SCD decomposition
        return decompose_scd(
            data=data,
            fsamp=fsamp,
            algorithm_config=algorithm_config,
            engine=engine,
            container=container,
        )
    
    elif method == "cbss":

        # Call FastIcaCBSS method
        return decompose_cbss(
            data=data, 
            fsamp=fsamp,
            algorithm_config=algorithm_config
        )
    
    elif method == "ae":

        # Call AEDecomposer
        return decompose_ae(
            data=data,
            fsamp=fsamp,
            algorithm_config=algorithm_config
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: scd, cbss, ae"
        )

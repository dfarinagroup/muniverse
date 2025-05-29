"""
Benchmark algorithms for decomposition.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
from ..utils.containers import pull_container, verify_container_engine
from .decomposition import decompose_scd, decompose_upperbound, decompose_cbss

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
    method: str = "scd",
    algorithm_config: Optional[str] = None,
    output_dir: str = "outputs",
    engine: str = "singularity",
    container: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """
    Decompose EMG recordings using specified method.
    
    Args:
        data: Either a path to input data file (.npy or .edf) or a numpy array of EMG data
        method: Decomposition method to use ("scd", "upperbound", or "cbss")
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image (required for SCD method)
        cache_dir: Directory for caching (optional)
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Route to appropriate method
    if method == "scd":
        if container is None:
            raise ValueError("Container path must be provided for SCD method")
        return decompose_scd(
            data=data,  # Will be converted to .npy file if needed
            algorithm_config=algorithm_config,
            output_dir=output_dir,
            engine=engine,
            container=container,
            cache_dir=cache_dir
        )
    elif method in ["upperbound", "cbss"]:
        # For internal methods, convert to numpy array if needed
        if isinstance(data, str):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Input data file not found: {data_path}")
            if data_path.suffix not in ['.npy', '.edf']:
                raise ValueError(f"Unsupported file format: {data_path.suffix}. Must be .npy or .edf")
            
            # Load data into numpy array
            if data_path.suffix == '.edf':
                from edfio import read_edf
                raw = read_edf(data_path)
                n_channels = raw.num_signals
                data = np.stack(raw.signals[i] for i in range(n_channels))
            else:  # .npy
                data = np.load(data_path)
        
        # Validate numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be either a file path (str) or numpy array")
        if data.ndim != 2:
            raise ValueError("EMG data must be a 2D array (channels x samples)")
        
        # Call appropriate method
        if method == "upperbound":
            return decompose_upperbound(
                data=data,
                algorithm_config=algorithm_config,
                output_dir=output_dir
            )
        else:  # cbss
            return decompose_cbss(
                data=data,
                algorithm_config=algorithm_config,
                output_dir=output_dir
            )
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of: scd, upperbound, cbss") 
"""
High-level wrapper functions for EMG decomposition with logging.

These functions never raise exceptions. They always return (results, log_data),
even on failure. Failed decompositions return {"sources": None, ...} with logs containing
error details. Callers could check if results["sources"] is None to detect failures.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logging import AlgorithmLogger
from .cbss import CBSS
from .upperbound import UpperBound
from .ae_decomposer import AEDecoder, AEDecoderConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def decompose_scd(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    engine: str = "singularity",
    container: str = "environment/muniverse_scd.sif",
    metadata: Optional[Dict] = None,
    repo_path: Optional[str] = None,
    conda_env: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Run SCD decomposition using container.

    Args:
        data: numpy array of EMG data (channels x samples)
        algorithm_config: Optional dictionary containing algorithm configuration
        engine: Container engine to use ("docker", "singularity" or "host")
        container: Path to container image
        metadata: Optional dictionary containing input data metadata for logging
        repo_path: If SCD runs on your native OS provide the path to the folder hosting the code
        conda_env: If SCD runs on your native OS specify which conda environment to be used

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics (if available)
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
        
    if engine == "host":
        if not os.path.isdir(repo_path):
            raise ValueError(f"Invalid local repository path: {repo_path}")
        try:
            scd_git_info = logger._get_git_info(repo_path)
            logger.add_generated_by(
                name="Swarm Contrastive Decomposition",
                url=scd_git_info["URL"],
                commit=scd_git_info["Commit"],
                branch=scd_git_info["Branch"],
                license="Creative Commons Attribution-NonCommercial 4.0 International Public License",
            )
        except:
            raise ValueError(f"Failed to extract the local repository metadata")
            
    elif engine in ["docker", "singularity"]:
        logger.add_generated_by(
            name="Swarm Contrastive Decomposition",
            url="https://github.com/AgneGris/swarm-contrastive-decomposition.git",
            commit="632a9ad041cf957584926d6b5cc64b7fe741e9eb",
            license="Creative Commons Attribution-NonCommercial 4.0 International Public License",
        )
    else:
        raise ValueError(f"Invalid engine {engine}")

    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "scd.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(
                f"Default SCD config not found at {algorithm_config}"
            )
        algo_cfg = load_config(algorithm_config)
        logger.set_algorithm_config(algo_cfg)

    # Create single run directory following neuromotion pattern
    with tempfile.TemporaryDirectory() as run_dir:
        run_dir = Path(run_dir)

        try:
            # Save data as standardized input file
            input_data_path = run_dir / "input_data.npy"
            np.save(input_data_path, data)
            
            # Save config as standardized config file (using already loaded config)
            config_path = run_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(algo_cfg, f, indent=2)

            # Get the absolute path to the script
            current_dir = Path(__file__).parent
            run_script_path = current_dir / "_run_scd.sh"
            script_path = current_dir / "_run_scd.py"

            if not run_script_path.exists():
                raise FileNotFoundError(f"Script not found at {run_script_path}")

            # Build container command with unified run_dir
            cmd = [
                str(run_script_path),
                engine,
                container,
                str(script_path),
                str(run_dir),
                repo_path,
                conda_env,
            ]

            # Run container
            subprocess.run(cmd, check=True, cwd=current_dir)
            print(f"[INFO] Decomposition completed successfully")
            logger.set_return_code("run.sh", 0)

            # Load results from container output files in run_dir
            results = {}
            
            # Load sources if available
            sources_path = run_dir / "predicted_sources.npz"
            if sources_path.exists():
                sources_data = np.load(sources_path)
                results["sources"] = sources_data["predicted_sources"]
            else:
                results["sources"] = None
            
            # Load spikes if available
            spikes_path = run_dir / "predicted_timestamps.tsv"
            if spikes_path.exists():
                spikes_df = pd.read_csv(spikes_path, sep="\t")
                # Convert back to dictionary format
                spikes_dict = {}
                for unit_id in spikes_df["unit_id"].unique():
                    unit_spikes = spikes_df[spikes_df["unit_id"] == unit_id]["timestamp"].values
                    spikes_dict[unit_id] = unit_spikes.tolist()
                results["spikes"] = spikes_dict
            else:
                results["spikes"] = {}
            
            # SCD doesn't typically provide silhouette scores
            # TODO: Compute silhouette scores
            results["silhouette"] = None
            
            # Log output files for tracking
            for root, _, files in os.walk(run_dir):
                for file in files:
                    if "input_data.npy" in file:
                        continue
                    file_path = os.path.join(root, file)
                    logger.add_output(file_path, os.path.getsize(file_path))

        except Exception as e:
            print(f"[ERROR] SCD decomposition failed: {str(e)}")
            logger.set_return_code("run.sh", 1)
            results = {"sources": None, "spikes": {}, "silhouette": None}

        finally:
            # Always finalize logger to ensure metadata is captured
            if engine == "host":
                logger.finalize()
            else:
                logger.finalize(engine, container)
        
        return results, logger.log_data


def decompose_upperbound(
    data: np.ndarray,
    muaps: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.

    Args:
        data: EMG data array (channels x samples)
        muaps: MUAPs array (n_motor_units x n_channels x duration)
        algorithm_config: Optional path to algorithm configuration JSON file
        metadata: Optional dictionary containing input data metadata for logging

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics
          * mu_filters: Motor unit filters
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    
    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        # Handle nested Config structure if present
        if "Config" in algorithm_config:
            algo_cfg = algorithm_config["Config"]
        else:
            # Assume the dict is the config itself
            algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config_path = config_dir / "upperbound.json"
        if not algorithm_config_path.exists():
            raise FileNotFoundError(
                f"Default UpperBound config not found at {algorithm_config_path}"
            )
        algo_cfg = load_config(str(algorithm_config_path))["Config"]
        logger.set_algorithm_config(algo_cfg)

    # Get sampling frequency from config
    fsamp = algo_cfg.get("sampling_frequency", 2048)

    try:
        # Initialize and run upperbound
        # Apply start and end time to data
        start_time = algo_cfg["start_time"] * algo_cfg["sampling_frequency"]
        end_time = algo_cfg["end_time"] * algo_cfg["sampling_frequency"]
        data = data[:, start_time:end_time].copy()
        
        ub = UpperBound(config=SimpleNamespace(**algo_cfg))

        # Validate muaps format
        if muaps.ndim != 3:
            raise ValueError("MUAPs must be a 3D array (n_motor_units x n_channels x duration)")

        # Run decomposition
        sources, spikes, sil, mu_filters = ub.decompose(data, muaps, fsamp=fsamp)

        # Prepare results
        results = {
            "sources": sources,
            "spikes": spikes,
            "silhouette": sil,
            "mu_filters": mu_filters,
        }

        logger.set_return_code("upperbound", 0)
        print(f"[INFO] UpperBound decomposition completed successfully")

    except Exception as e:
        print(f"[ERROR] UpperBound decomposition failed: {str(e)}")
        logger.set_return_code("upperbound", 1)
        results = {"sources": None, "spikes": {}, "silhouette": None, "mu_filters": None}
    
    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()
    
    return results, logger.log_data


def decompose_cbss(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run CBSS decomposition.

    Args:
        data: numpy array of EMG data (channels x samples)
        algorithm_config: Optional path to algorithm configuration JSON file
        metadata: Optional dictionary containing input data metadata for logging

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()

    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        # Handle nested Config structure if present
        if "Config" in algorithm_config:
            algo_cfg = algorithm_config["Config"]
        else:
            # Assume the dict is the config itself
            algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "cbss.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(
                f"Default CBSS config not found at {algorithm_config}"
            )
        algo_cfg = load_config(algorithm_config)["Config"]
        logger.set_algorithm_config(algo_cfg)

    try:
        # Apply start and end time to data
        start_time = algo_cfg["start_time"] * algo_cfg["sampling_frequency"]
        end_time = algo_cfg["end_time"] * algo_cfg["sampling_frequency"]
        data = data[:, int(start_time):int(end_time)]

        # Initialize and run CBSS with config
        cbss = CBSS(config=SimpleNamespace(**algo_cfg))
        sources, spikes, sil, _ = cbss.decompose(
            data, fsamp=algo_cfg["sampling_frequency"]
        )

        # Prepare results
        results = {"sources": sources, "spikes": spikes, "silhouette": sil}

        print(f"[INFO] CBSS decomposition completed successfully")
        logger.set_return_code("cbss", 0)

    except Exception as e:
        print(f"[ERROR] CBSS decomposition failed: {str(e)}")
        logger.set_return_code("cbss", 1)
        results = {"sources": None, "spikes": {}, "silhouette": None}
    
    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()
        
    return results, logger.log_data

def decompose_ae(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run Autoencoder-based decomposition.

    Args:
        data: EMG data (channels x samples)
        algorithm_config: Optional dict with "Config" key or the raw config dict
        metadata: Optional dict for logging (e.g., {"filename": "...", "format": "..."})

    Returns:
        results dict with:
            - sources: (n_units x n_samples)
            - spikes:  {unit_id: np.ndarray of sample indices}
            - silhouette: np.ndarray
            - mu_filters: (n_units x (m*R))
        and the logger data dict
    """
    logger = AlgorithmLogger()

    # Input data metadata
    if metadata:
        logger.set_input_data(
            file_name=metadata.get("filename", "numpy_array"),
            file_format=metadata.get("format", "npy"),
        )
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load config
    if algorithm_config:
        algo_cfg = algorithm_config.get("Config", algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        config_path = config_dir / "ae_decomposer.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Default AE config not found at {config_path}")
        algo_cfg = load_config(str(config_path))["Config"]
        logger.set_algorithm_config(algo_cfg)

    try:
        # Build strongly-typed config directly (no key remapping)
        ae_cfg = AEDecoderConfig(**algo_cfg)

        # Slice time window using config (seconds)
        fsamp = float(ae_cfg.sampling_frequency)
        start_idx = int(round(ae_cfg.start_time * fsamp))
        end_idx = int(round(ae_cfg.end_time * fsamp))
        data = data[:, start_idx:end_idx].copy()

        # Run AE
        ae = AEDecoder(ae_cfg)
        sources, spikes, sil, mu_filters = ae.decompose(data, fsamp=fsamp)

        results = {
            "sources": sources,
            "spikes": spikes,
            "silhouette": sil,
            "mu_filters": mu_filters,
        }

        logger.set_return_code("ae_decomposer", 0)
        print("[INFO] AE decomposition completed successfully")

    except Exception as e:
        print(f"[ERROR] AE decomposition failed: {e}")
        logger.set_return_code("ae_decomposer", 1)
        results = {"sources": None, "spikes": {}, "silhouette": None, "mu_filters": None}

    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()
    
    return results, logger.log_data

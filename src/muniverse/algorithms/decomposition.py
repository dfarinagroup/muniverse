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
from typing import Any, Dict, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

from ..utils.logging import AlgorithmLogger
from .cbss import FastIcaCBSS
from .upperbound import UpperBoundCBSS
from .ae_decomposer import AEDecoder #, AEDecoderConfig
from .pre_processing import PreProcessEMG
from .post_processing import PostProcessSpikes
from .core import map_spikes, spike_dict_to_long_df


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)
    
def _segment_data(data, ch_mask, sample_mask):
    """Helper function to segment data """

    segmented = data

    if ch_mask is not None:
        segmented = data[ch_mask, :]
    if sample_mask is not None:    
        segmented = segmented[:, sample_mask]

    return segmented

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
    logger.log_data["Pipeline"]["Name"] = "MUniverse-SCD-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Motor unit identification algorithm"
        
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
            container=logger._get_container_info(engine, container),
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
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        algorithm_config = config_dir / "scd.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(
                f"Default SCD config not found at {algorithm_config}"
            )
        algo_cfg = load_config(str(algorithm_config))
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

    Args
    ----
        data : np.ndarray (n_channels, n_samples)
            EMG data 
        muaps : np.ndarray (n_units, n_channels, n_samples)
            Impulse response waveforms for each motor unit    
        algorithm_config : dict (Optional) 
            Dictonary with the pipeline configuration
        metadata: dict (Optional)
            Optional dictionary containing input data metadata for logging

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
        log : dict
            Dictonary of processing metadata        
        model : UpperBoundCBSS
            The model used for decomposition    
    """
    # Initialize logger
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-UpperBoundCBSS-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Upper bound prediction for linear motor unit identification algorithms"
    
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
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        algorithm_config_path = config_dir / "upperbound.json"
        if not algorithm_config_path.exists():
            raise FileNotFoundError(
                f"Default UpperBound config not found at {algorithm_config_path}"
            )
        algo_cfg = load_config(str(algorithm_config_path))
        logger.set_algorithm_config(algo_cfg)

    try:

        # Validate muaps format
        if muaps.ndim != 3:
            raise ValueError(
                "MUAPs must be a 3D array (n_motor_units x n_channels x duration)"
            )
        
        # Apply preprocessing steps
        if "preProcessingConfig" in algo_cfg.keys():
            pre_module = PreProcessEMG(steps=algo_cfg["preProcessingConfig"])
            data, pre_meta = pre_module.pre_process(
                data=data, fsamp=algo_cfg["sampling_frequency"]
            )

            for step in pre_meta["steps"]:
                logger.add_processing_step(
                    step_name=step["step"],
                    details=step
                )
            
            # Get the data segment relevant for decomposition
            segmented_data = _segment_data(
                data, pre_meta["ch_mask"], pre_meta["sample_mask"]
            )
        else:
            segmented_data = data

        # TODO Apply any filtering operation also to the MUAPs (if used)
     
        # Initialize and run upperbound
        model = UpperBoundCBSS(config=SimpleNamespace(**algo_cfg))
        # Run decomposition
        spikes, segmented_sources, scores = model.fit_predict(
            sig=segmented_data, fsamp=pre_meta["fsamp"]
        )

        if "preProcessingConfig" in algo_cfg.keys():
        # Map spikes and sources to gloabl time
            sources = np.zeros((segmented_sources.shape[0], data.shape[1]))
            sources[:, pre_meta["sample_mask"]] = segmented_sources
            if pre_meta["t_start"] > 0:
                spikes = map_spikes(spikes, pre_meta["fsamp"], pre_meta["t_start"])

        # Apply post processing
        if "postProcessingConfig" in algo_cfg.keys():
            post_module = PostProcessSpikes(steps=algo_cfg["postProcessingConfig"])
            spikes, sources, scores, post_meta = post_module.post_process(
                spikes=spikes, 
                fsamp=pre_meta["fsamp"], 
                scores=scores, 
                sources=sources
            )

            for step in post_meta["steps"]:
                logger.add_processing_step(
                    step_name=step["step"],
                    details=step
                )

        # Prepare results
        results = {
            "data": data,
            "sources": sources, 
            "spikes": spikes, 
            "scores": scores
        }
        if "preProcessingConfig" in algo_cfg.keys():
            results["pre_process_meta"] = pre_meta
        if "postProcessingConfig" in algo_cfg.keys():
            results["post_process_meta"] = post_meta

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
) -> Tuple[Dict, Dict, FastIcaCBSS]:
    """
    Run CBSS decomposition.

    Args
    ----
        data : np.ndarray 
            EMG data (n_channels, n_samples)
        algorithm_config : dict (Optional) 
            Dictonary with the pipeline configuration
        metadata: dict (Optional)
            Optional dictionary containing input data metadata for logging

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
        log : dict
            Dictonary of processing metadata        
        model : FastIcaCBSS
            The model used for decomposition

    """
    # Initialize logger
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-CBSS-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Motor unit identification algorithm"

    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    algo_cfg = _get_config(algorithm_config, "cbss")
    logger.set_algorithm_config(algo_cfg)

    # Get the sampling rate
    fsamp = algo_cfg["sampling_frequency"]

    # Run preprocessing module
    if "preProcessingConfig" in algo_cfg.keys():
        steps = algo_cfg["preProcessingConfig"]
    else:
        steps = []
    data, segmented_data, pre_meta, return_code = _pre_process_data(data, steps, fsamp)  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        for step in pre_meta["steps"]:
            logger.add_processing_step(
                step_name="PreProcessing",
                details=step
            )    

    # Run CBSS decomposition
    spikes, seg_sources, scores, model, return_code = _run_cbss(
        segmented_data, 
        pre_meta["fsamp"], 
        SimpleNamespace(**algo_cfg["algorithmConfig"])
    )  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        logger.add_processing_step(
            step_name="FastIcaCBSS",
            details=algo_cfg["algorithmConfig"]
        )

    # Apply post processing
    if seg_sources is not None:
        # Map spikes and sources to gloabl time
        sources = np.zeros((seg_sources.shape[0], data.shape[1]))
        sources[:, pre_meta["sample_mask"]] = seg_sources

    if pre_meta["t_start"] > 0:
        spikes = map_spikes(spikes, pre_meta["fsamp"], pre_meta["t_start"])

    if "postProcessingConfig" in algo_cfg.keys():
        steps = algo_cfg["postProcessingConfig"]   
    else:
        steps = []

    spikes, sources, scores, post_meta, return_code = _post_process_spikes(
        spikes, sources, scores, pre_meta["fsamp"], steps
    )
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        for step in pre_meta["steps"]:
            logger.add_processing_step(
                step_name="PostProcessing",
                details=step
            )    


        # Prepare results
        results = {
            "data": data,
            "sources": sources, 
            "spikes": spikes, 
            "scores": scores,
            "pre_process_meta": pre_meta,
            "post_process_meta": post_meta
        }

        # Always finalize logger to ensure metadata is captured
        logger.finalize()
        
    return results, logger.log_data, model

def decompose_ae(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run Autoencoder-based decomposition.

    Args
    ----
        data : np.ndarray 
            EMG data (n_channels, n_samples)
        algorithm_config : dict (Optional) 
            Dictonary with the pipeline configuration
        metadata: dict (Optional)
            Optional dictionary containing input data metadata for logging

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
        log : dict
            Dictonary of processing metadata        
        model : AEDecoder
            The model used for decomposition

    """
    
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-AE-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Motor unit identification algorithm"

    # Input data metadata
    if metadata:
        logger.set_input_data(
            file_name=metadata.get("filename", "numpy_array"),
            file_format=metadata.get("format", "npy"),
        )
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    algo_cfg = _get_config(algorithm_config, "ae")
    logger.set_algorithm_config(algo_cfg)

    # Get the sampling rate
    fsamp = algo_cfg["sampling_frequency"]

    # Run preprocessing module
    if "preProcessingConfig" in algo_cfg.keys():
        steps = algo_cfg["preProcessingConfig"]
    else:
        steps = []
    data, segmented_data, pre_meta, return_code = _pre_process_data(data, steps, fsamp)  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        for step in pre_meta["steps"]:
            logger.add_processing_step(
                step_name="PreProcessing",
                details=step
            )    

    # Run CBSS decomposition
    spikes, seg_sources, scores, model, return_code = _run_ae(
        segmented_data, 
        pre_meta["fsamp"], 
        SimpleNamespace(**algo_cfg["algorithmConfig"])
    )  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        logger.add_processing_step(
            step_name="FastIcaCBSS",
            details=algo_cfg["algorithmConfig"]
        )

    # Apply post processing
    if seg_sources is not None:
        # Map spikes and sources to gloabl time
        sources = np.zeros((seg_sources.shape[0], data.shape[1]))
        sources[:, pre_meta["sample_mask"]] = seg_sources

    if pre_meta["t_start"] > 0:
        spikes = map_spikes(spikes, pre_meta["fsamp"], pre_meta["t_start"])

    if "postProcessingConfig" in algo_cfg.keys():
        steps = algo_cfg["postProcessingConfig"]   
    else:
        steps = []

    spikes, sources, scores, post_meta, return_code = _post_process_spikes(
        spikes, sources, scores, pre_meta["fsamp"], steps
    )
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        for step in pre_meta["steps"]:
            logger.add_processing_step(
                step_name="PostProcessing",
                details=step
            )    


        # Prepare results
        results = {
            "data": data,
            "sources": sources, 
            "spikes": spikes, 
            "scores": scores,
            "pre_process_meta": pre_meta,
            "post_process_meta": post_meta
        }

        # Always finalize logger to ensure metadata is captured
        logger.finalize()
    
    return results, logger.log_data, model

def _get_config(cfg, method):

    if isinstance(cfg, dict):
        algo_cfg = cfg
    elif isinstance(cfg, str):
        algo_cfg = load_config(str(cfg))
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        default_config = config_dir / f"{method}.json"
        if not default_config.exists():
            raise FileNotFoundError(
                f"Default SCD config not found at {default_config}"
            )
        algo_cfg = load_config(str(default_config))

    return algo_cfg

def _run_scd_local(data, cfg):
    """Run SCD decomposition locally"""

    try:
        import torch
        from scd import SwarmContrastiveDecomposition
        from scd.config.structures import Config, set_random_seed

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["device"] = device

        config = Config(**cfg)

        seed = cfg.get("Seed", 42)
        set_random_seed(seed=seed)

        neural_data = torch.from_numpy(data.T).to(
            device=device, dtype=torch.float32
        )

        model = SwarmContrastiveDecomposition()
        predicted_timestamps, dictionary = model.run(neural_data, config)

        return_code = {
                "name": "scd", 
                "value": 0
            }
        
        results = {} # TODO

    except Exception as e:
            print(f"[ERROR] SCD decomposition failed: {str(e)}")
            return_code = {
                "name": "scd", 
                "value": 1
            }
            results = {"sources": None, "spikes": {}, "silhouette": None}

    return results, return_code         
    
def _run_scd_container(data, algo_cfg, engine, container):
    """Run SCD decomposition in container"""

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
            ]

            # Run container
            subprocess.run(cmd, check=True, cwd=current_dir)
            print(f"[INFO] Decomposition completed successfully")
            return_code = {
                "name": "run_scd.sh", 
                "value": 0
            }

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

        except Exception as e:
            print(f"[ERROR] SCD decomposition failed: {str(e)}")
            return_code = {
                "name": "run_scd.sh", 
                "value": 0
            }
            results = {"sources": None, "spikes": {}, "silhouette": None}       


def _run_cbss(data, fsamp, cfg):
    """Run CBSS decomposition"""

    try:
        model = FastIcaCBSS(config=cfg)

        spikes, sources, scores = model.fit_predict(
                sig=data, fsamp=fsamp
            )
        
        return_code = {
            "name": "FastIcaCBSS", 
            "value": 0
        } 

    except Exception as e:
        print(f"[ERROR] FastIcaCBSS failed: {str(e)}")
        return_code = {
            "name": "FastIcaCBSS", 
            "value": 1
        } 
        spikes = None
        sources = None
        scores = None
        model = None   
        
    return spikes, sources, scores, model, return_code

def _run_ae(data, fsamp, cfg):
    """Run autoencoder decomposition"""

    try:
        model = AEDecoder(config=cfg)

        spikes, sources, scores = model.fit_predict(
                sig=data, fsamp=fsamp
            )
        return_code = {
            "name": "AEDecoder", 
            "value": 1
        } 
        
    except Exception as e:
        print(f"[ERROR] AEDecoder failed: {str(e)}")
        return_code = {
            "name": "AEDecoder", 
            "value": 1
        } 
        spikes = None
        sources = None
        scores = None
        model = None     

    return spikes, sources, scores, model, return_code

def _pre_process_data(data, steps, fsamp):
    """Preprocess EMG Data"""
    
    try:
    
        module = PreProcessEMG(steps=steps)
        data, pre_meta = module.pre_process(
            data=data, fsamp=fsamp
        )

        segmented_data = data

        if pre_meta["ch_mask"] is not None:
            segmented_data = segmented_data[pre_meta["ch_mask"], :]
        if pre_meta["sample_mask"] is not None:    
            segmented_data = segmented_data[:, pre_meta["sample_mask"]]    

        return_code = {
            "name": "PreProcessEMG", 
            "value": 0
        }    

    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {str(e)}")
        return_code = {
            "name": "PreProcessEMG", 
            "value": 1
        }
        segmented_data = data
        pre_meta = {
            "fsamp": fsamp,
            "ch_mask": np.ones(data.shape[0], dtype=bool),
            "sample_mask": np.ones(data.shape[1], dtype=bool),
            "steps": []
        }

    return data, segmented_data, pre_meta, return_code

def _post_process_spikes(spikes, sources, scores, fsamp, cfg):
    """Postprocess Spikes"""
    
    try:
    
        module = PostProcessSpikes(steps=cfg)
        spikes, sources, scores, post_meta = module.post_process(
            spikes=spikes, 
            fsamp=fsamp, 
            scores=scores, 
            sources=sources
        )   

        return_code = {
            "name": "PostProcessSpikes", 
            "value": 0
        }    

    except Exception as e:
        print(f"[ERROR] PostProcessSpikes failed: {str(e)}")
        return_code = {
            "name": "PostProcessSpikes", 
            "value": 1
        }
        post_meta = {
            "steps": []
        }

    return spikes, sources, scores, post_meta, return_code

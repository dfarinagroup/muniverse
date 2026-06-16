"""
High-level wrapper functions for EMG decomposition pipelines with logging.

These functions never raise exceptions. They always return (results, log_data),
even on failure. Failed decompositions return {"spikes": None, ...} with logs containing
error details. Callers could check if results["spikes"] is None to detect failures.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union, Literal
from importlib.metadata import metadata
from importlib.resources import files

import numpy as np
import pandas as pd
import pickle

from ..utils.logging import AlgorithmLogger
from .cbss import FastIcaCBSS
from .upperbound import UpperBoundCBSS
from .ae_decomposer import AEDecoder #, AEDecoderConfig
from .pre_processing import PreProcessEMG
from .post_processing import PostProcessSpikes
from .core import map_spikes, spike_dict_to_long_df

try:
    import scd as scd
    import torch
except ImportError:
    scd = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)
    
def decompose_scd(
    data: np.ndarray,
    fsamp: float,
    algorithm_config: Optional[Dict] = None,
    engine: Literal["local", "docker", "singularity"] = "singularity",
    container: str = "environment/muniverse_scd.sif",
    meta: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """
    API to run SCD decomposition using a container or a local installation.

    Args
    ----
        data : np.ndarray
            EMG data (n_channels, n_samples)

        fsamp : float
            Sampling rate in Hz

        algorithm_config : dict 
            Optional dictionary containing algorithm configuration

        engine : {"docker", "singularity", "local"} , default "singularity"
            Engine/container used to execute SCD. If "local" SCD is 
            executed using your local installation (pip installed).

        container : str
             Path to container image

        meta: dict 
            Optional dictionary containing metadata for loging


    Returns
    -------
        results : dict
            Dictonary with decomposition results containing
                - data (np.ndarray): Pre-processed data
                - spikes (pd.DataFrame): Table of motor unit spikes
                - sources (np.ndarray): Predicted sources
                - scores (dict): Source quality metrics
                - pre_process_metadata (dict): Metadata correspoding to
                pre processing steps (Optional)
                - post_process_metadata (dict): Metadata correspoding to
                post processing steps (Optional)

        log_data : dict
            Dictionary with processing metadata    

    References
    ----------
    .. [1] Grison et et al., "A Particle Swarm Optimised Independence Estimator 
           for Blind Source Separation of Neurophysiological Time Series",
           IEEE Transactions on Biomedical Engineering, 2024 
    .. [2] Grison et et al., "Unlocking the full potential of high-density surface EMG: 
           novel non-invasive high-yield motor unit decomposition",
           The Journal of Physiology, 2025        
    .. [3] Mamidanna et et al., "MUniverse: A Simulation and Benchmarking 
           Suite for Motor Unit Decomposition", The Thirty-ninth Annual 
           Conference on Neural Information Processing Systems 
           Datasets and Benchmarks Track, 2025         

    """
    # Initialize logger
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-SCD-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Motor unit identification algorithm"
        
    if engine == "local":
        if scd is None:
            raise ImportError(
                "The scd package is not installed locally."
                "If engine is 'local' scd needs to be installed or run it from a container."
            )
        logger.add_generated_by(
            name="Swarm Contrastive Decomposition",
            version=metadata("swarm-contrastive-decomposition")["version"],
            url="https://github.com/AgneGris/swarm-contrastive-decomposition.git",
            commit="n/a", 
            license="Creative Commons Attribution-NonCommercial 4.0 International Public License",
        )
  
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
    if meta:
        logger.set_input_data(file_name=meta["filename"], file_format=meta["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    algo_cfg = _get_config(algorithm_config, "scd")   
    logger.set_algorithm_config(algo_cfg) 

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

    algo_cfg["algorithmConfig"]["sampling_frequency"] = pre_meta["fsamp"]

    # Run SCD decomposition
    if engine == "local":
        spikes, seg_sources, scores, state, return_code = _run_scd_local(
            data=segmented_data, 
            cfg=algo_cfg["algorithmConfig"]
        )
    else:
        spikes, seg_sources, scores, state, return_code =_run_scd_container(
            data=segmented_data, 
            algo_cfg=algo_cfg["algorithmConfig"], 
            engine=engine, 
            container=container
        )
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        logger.add_processing_step(
            step_name="SCD",
            details=algo_cfg["algorithmConfig"]
        )    

    # Apply post processing
    if seg_sources is not None:
        # Map spikes and sources to gloabl time
        sources = np.zeros((seg_sources.shape[0], data.shape[1]))
        sources[:, pre_meta["sample_mask"]] = seg_sources
    else:
        sources = seg_sources    

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
        "model_state": state,
        "pre_process_meta": pre_meta,
        "post_process_meta": post_meta
    }   

    # Always finalize logger to ensure metadata is captured
    if engine == "local":
        logger.finalize()
    else:
        logger.finalize(engine, container)
        
    return results, logger.log_data


def decompose_upperbound(
    data: np.ndarray,
    muaps: np.ndarray,
    fsamp: float,
    algorithm_config: Optional[Dict] = None,
    meta: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.

    Args
    ----
        data : np.ndarray (n_channels, n_samples)
            EMG data 

        muaps : np.ndarray (n_units, n_channels, n_samples)
            Impulse response waveforms for each motor unit   

        fsamp : float
            Sampling rate in Hz     

        algorithm_config : dict (Optional) 
            Dictonary with the pipeline configuration

        meta: dict (Optional)
            Optional dictionary containing input data metadata for loging

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
 
    """
    # Initialize logger
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-UpperBoundCBSS-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Upper bound prediction for linear motor unit identification algorithms"
    
    # Set input data information
    if meta:
        logger.set_input_data(file_name=meta["filename"], file_format=meta["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        # config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        # algorithm_config_path = config_dir / "upperbound.json"
        config_path = files("muniverse").joinpath("configs/upperbound.json")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Default config file for upperbound method not found"
            )
        algo_cfg = load_config(str(config_path))
        logger.set_algorithm_config(algo_cfg)


    # Validate muaps format
    if muaps.ndim != 3:
        raise ValueError(
            "MUAPs must be a 3D array (n_units, n_channels, n_samples)"
            )
        
    # Run preprocessing module
    if "preProcessingConfig" in algo_cfg.keys():
        steps = algo_cfg["preProcessingConfig"]
    else:
        steps = []
    data, segmented_data, pre_meta, return_code = _pre_process_data(
        data=data, 
        steps=steps, 
        fsamp=fsamp
    )  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        for step in pre_meta["steps"]:
            logger.add_processing_step(
                step_name="PreProcessing",
                details=step
            ) 

        # If the EMG was filtered, apply the same operations to the MUAPs
        STEPS = {"bandpass", "highpass", "lowpass", "notch"}
        filtered_steps = [d for d in pre_meta["steps"] if d.get("step") in STEPS]

        if len(filtered_steps) > 0:
            # Pad MUAPs with zeros to avoid edge effects
            pad_len = 100
            filt_muaps = np.pad(muaps, ((0,0), (0,0), (pad_len, pad_len)))
            for i in range(muaps.shape[0]):
                filt_muaps[i, : , :], _, _, _ = _pre_process_data(
                    data=filt_muaps[i, :, :], 
                    steps=filtered_steps, 
                    fsamp=fsamp
                )
            muaps = filt_muaps[:, :, pad_len:-pad_len]

        # If the data was downsampled, apply the same downsampling to the MUAPs
        if pre_meta["fsamp"] != fsamp:
            downsample = fsamp // pre_meta["fsamp"]
            muaps = muaps[:, :, ::downsample]

        # Apply the channel mask
        muaps = muaps[:, pre_meta["ch_mask"], :]            

     
    # Run UpperBound decomposition
    spikes, seg_sources, scores, state, return_code = _run_upper_bound(
        data=segmented_data, 
        muaps=muaps,
        fsamp=pre_meta["fsamp"], 
        cfg=SimpleNamespace(**algo_cfg["algorithmConfig"])
    ) 
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        logger.add_processing_step(
            step_name="UpperBoundCBSS",
            details=state["parameters"]
        ) 

    # Apply post processing
    if seg_sources is not None:
        # Map spikes and sources to gloabl time
        sources = np.zeros((seg_sources.shape[0], data.shape[1]))
        sources[:, pre_meta["sample_mask"]] = seg_sources
    else:
        sources = seg_sources

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
        "model_state": state,
        "pre_process_meta": pre_meta,
        "post_process_meta": post_meta
    }    

    # Finalize logger to ensure metadata is captured
    logger.finalize()
    
    return results, logger.log_data


def decompose_cbss(
    data: np.ndarray,
    fsamp: float,
    algorithm_config: Optional[Dict] = None,
    meta: Optional[Dict] = None,
) -> Tuple[Dict, Dict, FastIcaCBSS]:
    """
    API to run a CBSS decomposition pipeline with optional
    pre and post processing steps.

    Args
    ----
        data : np.ndarray 
            EMG data (n_channels, n_samples)

        fsamp : float
            Sampling rate in Hz     

        algorithm_config : dict (Optional) 
            Dictonary with the pipeline configuration

        meta: dict (Optional)
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

        log_data : dict
            Dictonary of processing metadata        


    """
    # Initialize logger
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-CBSS-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Motor unit identification algorithm"

    # Set input data information
    if meta:
        logger.set_input_data(file_name=meta["filename"], file_format=meta["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    algo_cfg = _get_config(algorithm_config, "cbss")
    logger.set_algorithm_config(algo_cfg)

    # Run preprocessing module
    if "preProcessingConfig" in algo_cfg.keys():
        steps = algo_cfg["preProcessingConfig"]
    else:
        steps = []
    data, segmented_data, pre_meta, return_code = _pre_process_data(
        data=data, 
        steps=steps, 
        fsamp=fsamp
    )  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        for step in pre_meta["steps"]:
            logger.add_processing_step(
                step_name="PreProcessing",
                details=step
            )    

    # Run CBSS decomposition
    spikes, seg_sources, scores, state, return_code = _run_cbss(
        data=segmented_data, 
        fsamp=pre_meta["fsamp"], 
        cfg=SimpleNamespace(**algo_cfg["algorithmConfig"])
    )  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        logger.add_processing_step(
            step_name="FastIcaCBSS",
            details=state["parameters"]
        )

    # Apply post processing
    if seg_sources is not None:
        # Map spikes and sources to gloabl time
        sources = np.zeros((seg_sources.shape[0], data.shape[1]))
        sources[:, pre_meta["sample_mask"]] = seg_sources
    else:
        sources = seg_sources

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
        "model_state": state,
        "pre_process_meta": pre_meta,
        "post_process_meta": post_meta
    }

    # Always finalize logger to ensure metadata is captured
    logger.finalize()
        
    return results, logger.log_data

def decompose_ae(
    data: np.ndarray,
    fsamp: float,
    algorithm_config: Optional[Dict] = None,
    meta: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run Autoencoder-based decomposition.

    Args
    ----
        data : np.ndarray 
            EMG data (n_channels, n_samples)

        fsamp: float
            Sampling rate in Hz

        algorithm_config : dict (Optional) 
            Dictonary with the pipeline configuration

        meta: dict (Optional)
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

        log_data : dict
            Dictonary of processing metadata        


    """
    
    logger = AlgorithmLogger()
    logger.log_data["Pipeline"]["Name"] = "MUniverse-AE-Pipeline"
    logger.log_data["Pipeline"]["Description"] = "Motor unit identification algorithm"

    # Input data metadata
    if meta:
        logger.set_input_data(
            file_name=meta.get("filename", "numpy_array"),
            file_format=meta.get("format", "npy"),
        )
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    algo_cfg = _get_config(algorithm_config, "ae")
    logger.set_algorithm_config(algo_cfg)

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

    # Run AE decomposition
    spikes, seg_sources, scores, state, return_code = _run_ae(
        segmented_data, 
        pre_meta["fsamp"], 
        SimpleNamespace(**algo_cfg["algorithmConfig"])
    )  
    # Update the logger
    logger.set_return_code(return_code["name"], return_code["value"])
    if return_code["value"] == 0:
        logger.add_processing_step(
            step_name="AEDecoder",
            details=state["parameters"]
        )

    # Apply post processing
    if seg_sources is not None:
        # Map spikes and sources to gloabl time
        sources = np.zeros((seg_sources.shape[0], data.shape[1]))
        sources[:, pre_meta["sample_mask"]] = seg_sources
    else:
        sources = seg_sources    

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
        "model_state": state,
        "pre_process_meta": pre_meta,
        "post_process_meta": post_meta
    }

    # Always finalize logger to ensure metadata is captured
    logger.finalize()
    
    return results, logger.log_data

def _get_config(cfg, method):

    if isinstance(cfg, dict):
        algo_cfg = cfg
    elif isinstance(cfg, str):
        algo_cfg = load_config(str(cfg))
    else:
        # Load default configuration
        # config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        # default_config = config_dir / f"{method}.json"
        config_path = files("muniverse").joinpath(f"configs/{method}.json")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Default config file for method = {method} not found"
            )
        algo_cfg = load_config(str(config_path))

    return algo_cfg

def _run_scd_local(data, cfg):
    """Run SCD decomposition locally"""

    try:
        # Configure SCD 
        fsamp = cfg["sampling_frequency"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["device"] = device
        config = scd.Config(**cfg)
        seed = cfg.get("Seed", 42)
        scd.set_random_seed(seed=seed)

        # Convert data to torch
        neural_data = torch.from_numpy(data.T).to(
            device=device, dtype=torch.float32
        )

        # Build model and run decomposition
        model = scd.SwarmContrastiveDecomposition()
        _, dictionary = model.run(neural_data, config)

        # Extract the output
        n_units = len(dictionary["timestamps"])
        spike_dict = {
            i: dictionary["timestamps"][i].tolist() for i in range(n_units)}
        spikes = spike_dict_to_long_df(spike_dict, fsamp)

        if n_units > 0:
            sources = np.hstack(dictionary["source"]).T

            scores = {
                "sil": torch.stack(
                    dictionary["silhouettes"]).cpu().numpy().astype(float),
                "cov_isi": torch.stack(
                    dictionary["cov"]).cpu().numpy().astype(float) 
            }
        else:
            sources = None
            scores = None

        return_code = {
                "name": "scd", 
                "value": 0
            }
        
        state = dictionary

    except Exception as e:
            print(f"[ERROR] SCD decomposition failed: {str(e)}")
            return_code = {
                "name": "scd", 
                "value": 1
            }
            spikes = None
            sources = None
            scores = None
            state = None

    return spikes, sources, scores, state, return_code         
    
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
        
            # Load sources if available
            sources_path = run_dir / "predicted_sources.npz"
            if sources_path.exists():
                sources_data = np.load(sources_path)
                sources = sources_data["predicted_sources"]
            else:
                sources = None
            
            # Load spikes if available
            spikes_path = run_dir / "predicted_timestamps.tsv"
            if spikes_path.exists():
                spikes = pd.read_csv(spikes_path, sep="\t")
            else:
                spikes = None

            # Load spikes if available
            scores_path = run_dir / "predicted_scores.npz"
            if scores_path.exists():
                scores_data = np.load(scores_path, allow_pickle=True)
                scores = {
                    "sil": scores_data["sil"],
                    "cov_isi": scores_data["cov_isi"]
                }
            else:
                scores = None 

            # Load the dictonary of the model state
            dict_path = run_dir / "state.pkl"
            if dict_path.exists():
                with open(dict_path, "rb") as f:
                    state = pickle.load(f) 
            else:
                state = None   

        except Exception as e:
            print(f"[ERROR] SCD decomposition failed: {str(e)}")
            return_code = {
                "name": "run_scd.sh", 
                "value": 1
            }
            spikes = None
            sources = None
            scores = None
            state = None

        return spikes, sources, scores, state, return_code           

def _run_upper_bound(data, muaps, fsamp, cfg):
    """Run UpperBoundCBSS decomposition"""

    try:
        model = UpperBoundCBSS(config=cfg)

        spikes, sources, scores = model.fit_predict(
                sig=data, muaps=muaps, fsamp=fsamp
            )
        
        state = model.save_model()
        
        return_code = {
            "name": "UpperBoundCBSS", 
            "value": 0
        } 

    except Exception as e:
        print(f"[ERROR] UpperBoundCBSS failed: {str(e)}")
        return_code = {
            "name": "UpperBoundCBSS", 
            "value": 1
        } 
        spikes = None
        sources = None
        scores = None 
        state = None
        
    return spikes, sources, scores, state, return_code


def _run_cbss(data, fsamp, cfg):
    """Run CBSS decomposition"""

    try:
        model = FastIcaCBSS(config=cfg)

        spikes, sources, scores = model.fit_predict(
                sig=data, fsamp=fsamp
            )
        
        state = model.save_model()
        
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
        state = None
        
    return spikes, sources, scores, state, return_code

def _run_ae(data, fsamp, cfg):
    """Run autoencoder decomposition"""

    try:
        model = AEDecoder(config=cfg)

        spikes, sources, scores = model.fit_predict(
                sig=data, fsamp=fsamp
            )
        
        state = model.save_model()

        return_code = {
            "name": "AEDecoder", 
            "value": 0
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
        state = None

    return spikes, sources, scores, state, return_code

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
            "steps": [],
            "t_start": 0
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

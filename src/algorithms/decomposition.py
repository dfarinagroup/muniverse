import json
import subprocess
from pathlib import Path
import tempfile
from edfio import read_edf
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union
import os
import time
from types import SimpleNamespace


from .decomposition_methods import upper_bound, basic_cBSS
from .decomposition_routines import spike_dict_to_long_df
from ..utils.logging import AlgorithmLogger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def decompose_scd(
    data: Union[str, np.ndarray],
    output_dir: Path,
    algorithm_config: Optional[str] = None,
    engine: str = "singularity",
    container: str = "environment/muniverse_scd.sif",
) -> Tuple[Dict, Dict]:
    """
    Run SCD decomposition using container.
    
    Args:
        data: Either a path to input data file (.npy or .edf) or numpy array of EMG data
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image
        cache_dir: Optional directory for caching
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    
    # Set input data information
    if isinstance(data, str):
        data_path = Path(data)
        logger.set_input_data(
            file_name=data_path.name,
            file_format=data_path.suffix[1:]  # Remove the dot
        )
    else:
        logger.set_input_data(
            file_name="numpy_array",
            file_format="npy"
        )
    
    # Load and set algorithm configuration
    if algorithm_config:
        algo_cfg = load_config(algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "scd.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(f"Default SCD config not found at {algorithm_config}")
        algo_cfg = load_config(algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    
    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Convert input to .npy file if needed
        if isinstance(data, np.ndarray):
            temp_emg_path = temp_dir / "temp_emg.npy"
            np.save(temp_emg_path, data)
        elif isinstance(data, str):
            data_path = Path(data)
            if data_path.suffix == '.edf':
                # Load EDF and save as .npy
                raw = read_edf(data_path)
                # Read channels.tsv file to get channel information
                try:
                    channels_df = pd.read_csv(data_path.parent / f"{data_path.stem.replace('_emg', '')}_channels.tab", delimiter='\t')
                    n_channels = len(channels_df[channels_df['type'].str.startswith('EMG')])
                except FileNotFoundError:
                    n_channels = raw.num_signals
                emg_data = np.stack([raw.signals[i].data for i in range(n_channels)])
                temp_emg_path = temp_dir / "temp_emg.npy"
                np.save(temp_emg_path, emg_data)
            else:  # .npy file
                temp_emg_path = data_path
        else:
            raise TypeError("data must be either a file path (str) or numpy array")
        
        # Get the absolute path to the script
        current_dir = Path(__file__).parent
        run_script_path = current_dir / "_run_scd.sh"
        script_path = current_dir / "_run_scd.py"
        
        if not run_script_path.exists():
            raise FileNotFoundError(f"Script not found at {run_script_path}")

        # Add preprocessing step
        logger.add_processing_step("Preprocessing", {
            "InputFormat": data_path.suffix[1:] if isinstance(data, str) else "numpy_array",
            "OutputFormat": "npy",
            "Description": f"Convert input data from {data_path.suffix[1:]} to numpy format for processing"
        })

        # Build container command
        cmd = [
            str(run_script_path),
            engine,
            container,
            str(script_path),
            str(temp_emg_path),
            str(algorithm_config),
            str(run_dir)
        ]
        
        # Run container
        try:
            subprocess.run(cmd, check=True, cwd=current_dir)
            print(f"[INFO] Decomposition completed successfully at {run_dir}")
            logger.set_return_code("run.sh", 0)            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Decomposition failed: {e}")
            print(f"[ERROR] Command output: {e.output if hasattr(e, 'output') else 'No output'}")
            print(f"[ERROR] Command stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            logger.set_return_code("run.sh", e.returncode)
            raise
        
        print(f"[INFO] Results saved to {run_dir}")
        
        # Log output files
        for root, _, files in os.walk(run_dir):
            for file in files: 
                file_path = os.path.join(root, file)
                logger.add_output(file_path, os.path.getsize(file_path))
        
        # Finalize and save the log
        log_path = logger.finalize(run_dir)
        print(f"Run log saved to: {log_path}")
        
        return None

def decompose_upperbound(
    data: np.ndarray,
    data_generation_config: str,
    muap_cache_file: Optional[str],
    algorithm_config: Optional[str],
    output_dir: Path
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.
    
    Args:
        data: EMG data array (channels x samples)
        data_generation_config: output_config from data generation
        muaps_cache_file: File where MUAPs are saved 
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    
    # Set input data information
    logger.set_input_data(
        file_name="numpy_array",
        file_format="npy"
    )
    # Load algorithm config if provided, otherwise use defaults
    if algorithm_config:
        algo_cfg = load_config(algorithm_config)
    else:
        algo_cfg = {"ext_fact":8, "whitening_method":"ZCA", "cluster_method":'kmeans', 'whitening_reg':'auto'} # Will use defaults

    logger.set_algorithm_config(algo_cfg)
    
    # Add preprocessing step
    logger.add_processing_step("Preprocessing", {
        "InputFormat": "numpy_array",
        "Description": "Input data is already in numpy format"
    })

    # Initialize and run upperbound
    ub = upper_bound(config=SimpleNamespace(**algo_cfg))
    
    # Use the new load_muaps method to get the MUAPs
    muaps_reshaped, fsamp, angle = ub.load_muaps(data_generation_config, muap_cache_file)
    
    # Move EMG to Nchannels, Nsamples shape
    data = data.T 
    sources, spikes, sil = ub.decompose(data, muaps_reshaped, fsamp=fsamp)
    
    # Add decomposition step
    logger.add_processing_step("Decomposition", {
        "Method": "UpperBound",
        "Configuration": algo_cfg,
        "Description": "Run UpperBound algorithm on input data",
        "MuapFile": str(muap_cache_file),
        "AngleUsed": angle
    })
    
    # Prepare results
    results = {
        'sources': sources,
        'spikes': spikes,
        'silhouette': sil
    }
    
    # Save results
    save_decomposition_results(output_dir, results, {})
    
    # Log output files
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))
    
    # Finalize and save the log
    log_path = logger.finalize(output_dir)
    print(f"Run log saved to: {log_path}")
    
    return results, {}

def decompose_cbss(
    data: Union[str, np.ndarray],
    output_dir: Path,
    algorithm_config: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Run CBSS decomposition.
    
    Args:
        data: Either a path to input data file (.npy or .edf) or numpy array of EMG data
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with process metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    
    # Set input data information
    if isinstance(data, str):
        data_path = Path(data)
        logger.set_input_data(
            file_name=data_path.name,
            file_format=data_path.suffix[1:]  # Remove the dot
        )
    else:
        logger.set_input_data(
            file_name="numpy_array",
            file_format="npy"
        )
    
    # Load and set algorithm configuration
    if algorithm_config:
        algo_cfg = load_config(algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "cbss.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(f"Default CBSS config not found at {algorithm_config}")
        algo_cfg = load_config(algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    
    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Load and preprocess data
        if isinstance(data, str):
            data_path = Path(data)
            if data_path.suffix == '.edf':
                # Load EDF and extract EMG data
                raw = read_edf(data_path)
                try:
                    channels_df = pd.read_csv(data_path.parent / f"{data_path.stem.replace('_emg', '')}_channels.tab", delimiter='\t')
                    n_channels = len(channels_df[channels_df['type'].str.startswith('EMG')])
                except FileNotFoundError:
                    n_channels = raw.num_signals
                emg_data = np.stack([raw.signals[i].data for i in range(n_channels)])
            else:  # .npy file
                emg_data = np.load(data_path)
        else:
            emg_data = data

        # Add preprocessing step
        logger.add_processing_step("Preprocessing", {
            "InputFormat": data_path.suffix[1:] if isinstance(data, str) else "numpy_array",
            "Description": "Load and prepare EMG data for CBSS decomposition"
        })

        # Initialize and run CBSS with config
        cbss = basic_cBSS(config=algo_cfg)
        sources, spikes, sil, _ = cbss.decompose(emg_data, fsamp=2048) 
        
        # Add decomposition step
        logger.add_processing_step("Decomposition", {
            "Method": "CBSS",
            "Configuration": algo_cfg,
            "Description": "Run CBSS algorithm on input data"
        })
        
        # Prepare results
        results = {
            'sources': sources,
            'spikes': spikes,
            'silhouette': sil
        }
        
        # Save results in appropriate formats
        # 1. Save spikes as TSV in long format
        spikes_df = spike_dict_to_long_df(spikes, fsamp=2048)
        spikes_path = os.path.join(run_dir, 'predicted_timestamps.tsv')
        spikes_df.to_csv(spikes_path, sep='\t', index=False)
        
        # 2. Save sources as compressed NPZ
        sources_path = os.path.join(run_dir, 'predicted_sources.npz')
        np.savez_compressed(sources_path, sources=sources)
        
        # 3. Save silhouette scores as compressed NPZ
        sil_path = os.path.join(run_dir, 'silhouette.npz')
        np.savez_compressed(sil_path, silhouette=sil)
        
        print(f"[INFO] Decomposition completed successfully at {run_dir}")
        logger.set_return_code("cbss", 0)
        
        # Log output files
        for root, _, files in os.walk(run_dir):
            for file in files: 
                file_path = os.path.join(root, file)
                logger.add_output(file_path, os.path.getsize(file_path))
        
        # Finalize and save the log
        log_path = logger.finalize(run_dir)
        print(f"Run log saved to: {log_path}")
        
        return results, logger.log_data
            
    except Exception as e:
        print(f"[ERROR] Decomposition failed: {str(e)}")
        logger.set_return_code("cbss", 1)
        raise

def save_decomposition_results(output_dir: Path, results: Dict, metadata: Dict):
    """Save decomposition results and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "decomposition_results.pkl"
    metadata_path = output_dir / "processing_metadata.json"
    
    # Save results
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results_path, metadata_path
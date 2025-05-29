import json
import subprocess
from pathlib import Path
import tempfile
import numpy as np
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
    data: np.ndarray,
    output_dir: Path,
    algorithm_config: Optional[str] = None,
    engine: str = "singularity",
    container: str = "environment/muniverse_scd.sif",
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run SCD decomposition using container.
    
    Args:
        data: numpy array of EMG data (channels x samples)
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image
        metadata: Optional dictionary containing input data metadata for logging
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    # Add SCD generator info
    logger.add_generated_by(
        name="Swarm Contrastive Decomposition",
        url="https://github.com/AgneGris/swarm-contrastive-decomposition.git",
        commit="632a9ad041cf957584926d6b5cc64b7fe741e9eb",
        license="Creative Commons Attribution-NonCommercial 4.0 International Public License"
    )
    
    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata['filename'], file_format=metadata['format'])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")
    
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
        
        # Save data as .npy file
        temp_emg_path = temp_dir / "temp_emg.npy"
        np.save(temp_emg_path, data)
        
        # Get the absolute path to the script
        current_dir = Path(__file__).parent
        run_script_path = current_dir / "_run_scd.sh"
        script_path = current_dir / "_run_scd.py"
        
        if not run_script_path.exists():
            raise FileNotFoundError(f"Script not found at {run_script_path}")

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
        
        print(f"[INFO] Results saved to {run_dir}")
        
        # Log output files
        for root, _, files in os.walk(run_dir):
            for file in files: 
                file_path = os.path.join(root, file)
                logger.add_output(file_path, os.path.getsize(file_path))
        
        # Finalize and save the log
        log_path = logger.finalize(run_dir)
        print(f"Run log saved to: {log_path}")
        
        return None, logger.log_data

def decompose_upperbound(
    data: np.ndarray,
    output_dir: Path,
    simulation_config: str,  # Changed parameter name
    muap_cache_file: Optional[str] = None,
    algorithm_config: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.

    Args:
        data: EMG data array (channels x samples)
        output_dir: Directory to save results
        simulation_config: Path to simulation configuration JSON file
        muap_cache_file: Path to MUAP cache file
        algorithm_config: Optional path to algorithm configuration JSON file
        metadata: Optional dictionary containing input data metadata for logging

    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    print(metadata)
    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata['filename'], file_format=metadata['format'])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

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
    muaps_reshaped, fsamp, angle = ub.load_muaps(simulation_config, muap_cache_file)  # Updated method call
    
    # Add decomposition step
    logger.add_processing_step("Decomposition", {
        "Method": "UpperBound",
        "Configuration": algo_cfg,
        "Description": "Run UpperBound algorithm on input data",
        "MuapFile": str(muap_cache_file),
        "AngleUsed": angle,
        "SimulationConfig": str(simulation_config)  # Updated logging
    })

    # Move EMG to Nchannels, Nsamples shape
    sources, spikes, sil, mu_filters = ub.decompose(data, muaps_reshaped, fsamp=fsamp)  # Updated to receive mu_filters

    # Prepare results
    results = {
        'sources': sources,
        'spikes': spikes,
        'silhouette': sil,
        'mu_filters': mu_filters  # Include filters in results
    }

    logger.set_return_code("upperbound", 0)

    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Save results using the optimized format
    save_decomposition_results(run_dir, results, {}, fsamp=fsamp)
    # Log output files
    for root, _, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))
    # Finalize and save the log
    log_path = logger.finalize(run_dir)
    print(f"Run log saved to: {log_path}")

    return results, logger.log_data

def decompose_cbss(
    data: np.ndarray,
    output_dir: Path,
    algorithm_config: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run CBSS decomposition.
    
    Args:
        data: numpy array of EMG data (channels x samples)
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
        metadata: Optional dictionary containing input data metadata for logging
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with process metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    
    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata['filename'], file_format=metadata['format'])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")
    
    # Load and set algorithm configuration
    if algorithm_config:
        algo_cfg = load_config(algorithm_config)['Config']
        logger.set_algorithm_config(algo_cfg)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "cbss.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(f"Default CBSS config not found at {algorithm_config}")
        algo_cfg = load_config(algorithm_config)['Config']
        logger.set_algorithm_config(algo_cfg)
    
    # Create a unique run directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Initialize and run CBSS with config
        cbss = basic_cBSS(config=SimpleNamespace(**algo_cfg))
        sources, spikes, sil, _ = cbss.decompose(data, fsamp=algo_cfg['sampling_frequency']) 
        
        # Prepare results
        results = {
            'sources': sources,
            'spikes': spikes,
            'silhouette': sil
        }
        
        # Save results using the optimized format
        save_decomposition_results(run_dir, results, {}, fsamp=algo_cfg['sampling_frequency'])
        
        print(f"[INFO] Decomposition completed successfully at {run_dir}")
        logger.set_return_code("cbss", 0)
        
    except Exception as e:
        print(f"[ERROR] Decomposition failed: {str(e)}")
        logger.set_return_code("cbss", 1)
        results = None
    
    # Log output files
    for root, _, files in os.walk(run_dir):
        for file in files: 
            file_path = os.path.join(root, file)
            logger.add_output(file_path, os.path.getsize(file_path))
    
    # Finalize and save the log
    log_path = logger.finalize(run_dir)
    print(f"Run log saved to: {log_path}")
    
    return results, logger.log_data

def save_decomposition_results(output_dir: Path, results: Dict, metadata: Dict, fsamp: Optional[float] = None):
  """
  Save decomposition results and metadata in optimized formats.
  
  Args:
    output_dir: Directory to save results
    results: Dictionary of decomposition results containing sources, spikes, silhouette, etc.
    metadata: Dictionary with processing metadata
    fsamp: Sampling frequency, needed for converting spikes to time format
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  # 1. Save spikes as TSV in long format if fsamp is provided
  if fsamp is not None and 'spikes' in results:
    spikes_df = spike_dict_to_long_df(results['spikes'], fsamp=fsamp)
    spikes_path = output_dir / 'predicted_timestamps.tsv'
    spikes_df.to_csv(spikes_path, sep='\t', index=False)
  
  # 2. Save sources as compressed NPZ
  if 'sources' in results:
    sources_path = output_dir / 'predicted_sources.npz'
    np.savez_compressed(sources_path, sources=results['sources'])
  
  # 3. Save silhouette scores as compressed NPZ
  if 'silhouette' in results:
    sil_path = output_dir / 'silhouette.npz'
    np.savez_compressed(sil_path, silhouette=results['silhouette'])
  
  # 4. Save MU filters as compressed NPZ
  if 'mu_filters' in results:
    filters_path = output_dir / 'mu_filters.npz'
    np.savez_compressed(filters_path, mu_filters=results['mu_filters'])
  
  # 5. Save metadata as JSON (keeping this for compatibility)
  metadata_path = output_dir / "processing_metadata.json"
  with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
  
  return output_dir



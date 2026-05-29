#!/usr/bin/env python
"""
_run_scd.py - Container script for SCD decomposition

This script receives input data (EMG data, config) and returns SCD decomposition results.
Modified from https://github.com/AgneGris/swarm-contrastive-decomposition.git
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pickle

try:
    import torch
    from config.structures import Config, set_random_seed
    from models.scd import SwarmContrastiveDecomposition
    from processing.postprocess import save_results
except ImportError as e:
    print(f"Warning: Container-specific imports not available: {e}")
    print("This script is intended to run inside a container with SCD")

# Prepare the rows
def write_spike_tsv(dictionary, fsamp, output_dir):
    rows = []
    duration = 0
    desc = "motor-unit-spike"
    for unit_id, ts_tensor in enumerate(dictionary["timestamps"]):
        timestamps = ts_tensor.tolist()  # Convert tensor to list
        for ts in timestamps:
            onset = ts / fsamp
            rows.append(f"{onset}\t{duration}\t{ts}\t{unit_id}\t{desc}")
        
    rows_sorted = sorted(rows, key=lambda r: float(r.split("\t")[0])) 

    # Write to TSV file
    with open(output_dir / "predicted_timestamps.tsv", "w") as f:
        f.write("onest\tduration\tsample\tunit_id\tdescription\n")  # Write header
        for row in rows:
            f.write(f"{row}\n")

    return None

def write_sources(dictionary, output_dir):
    sources = np.hstack(dictionary["source"]).T
    np.savez_compressed(output_dir / "predicted_sources.npz", predicted_sources=sources)

    return None

def write_scores(dictionary, output_dir):
    try:
        scores = {
                "sil": torch.stack(
                    dictionary["silhouettes"]).cpu().numpy().astype(float),
                "cov_isi": torch.stack(
                    dictionary["cov"]).cpu().numpy().astype(float) 
            }
    except:
        scores = None   

    np.savez_compressed(output_dir / "predicted_scores.npz", **scores)

    return None

def write_dict(dictionary, output_dir):
    output_dir = Path(output_dir)
    try:
        with open(output_dir / "state.pkl", "wb") as f:
            pickle.dump(dictionary, f)
    except:
        pass


def train(run_dir: str):
    """Run SCD decomposition on EMG data."""
    # Load config from standardized location
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r") as f:
        alg_config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alg_config["device"] = device

    # Unpack config into Config dataclass
    config = Config(**alg_config)

    # Set random seed
    seed = alg_config.get("Seed", 42)
    set_random_seed(seed=seed)

    # Load data from standardized location
    data_path = os.path.join(run_dir, "input_data.npy")
    npy_data = np.load(data_path)
    d1, d2 = npy_data.shape
    print(f"Found {d1}x{d2} data")
    if d1 < d2:
        print(f"Transposing to {d2}x{d1}")
        npy_data = npy_data.T

    neural_data = torch.from_numpy(npy_data).to(
        device=config.device, dtype=torch.float32
    )

    # # Apply time window if specified
    # start_idx = int(config.start_time * config.sampling_frequency)
    # end_idx = int(config.end_time * config.sampling_frequency)
    # neural_data = neural_data[start_idx:end_idx, :]

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    # Save results to run directory
    run_path = Path(run_dir)
    
    write_spike_tsv(dictionary, alg_config["sampling_frequency"], run_path)
    write_sources(dictionary, run_path)
    write_scores(dictionary, run_path)
    write_dict(dictionary, run_dir)

    print(f"Saved results to {run_path}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCD decomposition on EMG data")
    parser.add_argument("--run_dir", type=str, required=True, 
                       help="Path to run directory containing input files and where output will be saved")
    
    args = parser.parse_args()
    train(args.run_dir)

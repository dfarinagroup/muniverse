# Modified from https://github.com/AgneGris/swarm-contrastive-decomposition.git

import sys
import numpy as np
import torch
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple, Any

from config.structures import set_random_seed, Config
from models.scd import SwarmContrastiveDecomposition
from processing.postprocess import save_results

# Prepare the rows
def write_spike_tsv(dictionary, output_dir):
    rows = []
    for unit_id, ts_tensor in enumerate(dictionary['timestamps']):
        timestamps = ts_tensor.tolist()  # Convert tensor to list
        for ts in timestamps:
            rows.append(f"{unit_id}\t{ts}")

    # Write to TSV file
    with open(output_dir / 'predicted_timestamps.tsv', 'w') as f:
        f.write("unit_id\ttimestamp\n")  # Write header
        for row in rows:
            f.write(f"{row}\n")

    return None

def write_sources(dictionary, output_dir):
    sources = np.hstack(dictionary['source'])
    np.savez_compressed(output_dir / 'predicted_sources.npz', predicted_sources=sources)

    return None

def train(data_path: str, config_path: str, output_dir: str):
    """Run SCD decomposition on EMG data."""
    # Load and set config
    with open(config_path, 'r') as f:
        alg_config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alg_config['Config']['device'] = device
    
    # Unpack config into Config dataclass
    config = Config(**alg_config['Config'])
    
    # Set random seed
    seed = alg_config.get('Seed', 42)
    set_random_seed(seed=seed)

    # Load data
    npy_data = np.load(data_path)
    d1, d2 = npy_data.shape
    print(f"Found {d1}x{d2} data")
    if d1 < d2:
        print(f"Transposing to {d2}x{d1}")
        npy_data = npy_data.T
    
    neural_data = torch.from_numpy(npy_data).to(device=config.device, dtype=torch.float32)
    
    # Apply time window if specified
    neural_data = neural_data[config.start_time * config.sampling_frequency : config.end_time * config.sampling_frequency, :]

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_spike_tsv(dictionary, output_dir)
    write_sources(dictionary, output_dir)
    print(f"Saved results to {output_dir}")
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SCD decomposition on EMG data')
    parser.add_argument('data_path', type=str, help='Path to .npy file containing EMG data')
    parser.add_argument('config_path', type=str, help='Path to algorithm configuration JSON file')
    parser.add_argument('output_dir', type=str, help='Directory to save results')
    
    args = parser.parse_args()
    
    train(args.data_path, args.config_path, args.output_dir)
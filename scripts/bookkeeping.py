#!/usr/bin/env python3
"""
Bookkeeping script to keep track of decomposition results for the MUniverse benchmark.

For each dataset and algorithm combination, creates and manages a CSV inventory
that tracks whether recordings have been decomposed successfully.

Usage:
    python bookkeeping.py -d <dataset> -a <algorithm>

Example:
    python bookkeeping.py -d Neuromotion-test -a scd

CLI Arguments:
    -d, --dataset: Name of the dataset to process
    -a, --algorithm: Algorithm to use for decomposition
    --create: Create a new inventory
    --update: Update the inventory status by checking output files
    --status: Print the status summary
    --bids-root: Override default BIDS root directory
    --output-dir: Override default output directory

"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Default paths
DATA_DIR = '/rds/general/user/pm1222/ephemeral/muniverse/data'
INTERIM_DIR = '/rds/general/user/pm1222/ephemeral/muniverse/interim'


def extract_bids_components(filename: str) -> Tuple[str, str]:
    """
    Extract BIDS components (subject, task) from filename.
    
    Args:
        filename: BIDS-formatted filename
        
    Returns:
        Tuple of (subject, task)
    """
    # Define patterns for each component
    patterns = {
        'subject': r'sub-(?:sim)?(\d+)',
        'task': r'task-(\w+)_run'
    }
    
    # Extract each component
    components = {}
    for component, pattern in patterns.items():
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(f"Could not find {component} in filename: {filename}")
        components[component] = match.group(1)
    
    return components['subject'], components['task']


def create_bids_dataframe(bids_root: Path) -> pd.DataFrame:
    """
    Create a DataFrame containing all EDF files in the BIDS structure.
    
    Args:
        bids_root: Path to BIDS root directory
        
    Returns:
        DataFrame with columns:
        - edf_path: Path to EDF file
        - log_config_path: Path to simulation config (if simulated data)
        - emg_sidecar_path: Path to EMG sidecar JSON (if experimental data)
        - subject: Subject ID
        - task: Task name
        - data_type: 'simulated' or 'experimental'
    """
    # Find all EDF files recursively
    edf_files = list(bids_root.rglob('*_emg.edf'))
    
    # Create lists to store the data
    data = []
    
    for edf_path in edf_files:
        try:
            # Extract BIDS components
            subject, task = extract_bids_components(edf_path.stem)
            
            # Determine if this is simulated or experimental data
            log_path = edf_path.parent / f"{edf_path.stem.replace('_emg', '')}_simulation.json"
            data_type = 'simulated' if log_path.exists() else 'experimental'
            
            data.append({
                'edf_path': str(edf_path),
                'subject': subject,
                'task': task,
                'data_type': data_type
            })
        except ValueError as e:
            print(f"Warning: Skipping file {edf_path.name} - {str(e)}")
            continue
    
    if not data:
        raise ValueError("No valid EDF files found in the BIDS structure")
    
    return pd.DataFrame(data)


def create_inventory(
    dataset_name: str,
    algorithm: str,
    bids_root: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create a new inventory CSV for a dataset-algorithm combination.
    
    Args:
        dataset_name: Name of the dataset
        algorithm: Algorithm name ('scd', 'cbss', 'upperbound')
        bids_root: Path to BIDS root (defaults to DATA_DIR/dataset_name)
        output_dir: Path to output directory (defaults to INTERIM_DIR/algorithm/dataset_name)
        
    Returns:
        DataFrame with inventory including status tracking columns
    """
    # Set default paths if not provided
    if bids_root is None:
        bids_root = Path(DATA_DIR) / dataset_name
        
    # Create base DataFrame from BIDS structure
    df = create_bids_dataframe(bids_root)
    
    # Add tracking columns
    df['recording_id'] = range(len(df))
    df['algorithm'] = algorithm
    df['status'] = 'pending'
    df['output_path'] = ' '
    df['last_updated'] = datetime.now().isoformat()
    df['error_message'] = ' '
    
    # Reorder columns for better readability
    columns = [
        'recording_id', 'subject', 'task', 'data_type', 'algorithm', 'status',
        'edf_path', 'output_path', 'last_updated', 'error_message'
    ]
    df = df[columns]
    
    # Save inventory
    inventory_path = output_dir / f'inventory_{algorithm}.csv'
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(inventory_path, index=False)
    
    print(f"Created inventory at: {inventory_path}")
    print(f"Total recordings: {len(df)}")
    print(f"Data types: {df['data_type'].value_counts().to_dict()}")
    
    return df


def update_inventory_status(
    inventory_path: Path,
    recording_id: Optional[int] = None,
    output_path: Optional[Path] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
    max_retries: int = 5,
    retry_delay: float = 0.5
) -> pd.DataFrame:
    """
    Update the status of one or all recordings in the inventory.
    Uses retry logic to handle concurrent access from multiple jobs.
    
    Args:
        inventory_path: Path to inventory CSV
        recording_id: ID of recording to update (None = update all)
        output_path: Path to output directory for recording
        status: New status ('pending', 'processing', 'completed', 'failed')
        error_message: Error message if status is 'failed'
        max_retries: Number of retry attempts for file access
        retry_delay: Delay between retries in seconds
        
    Returns:
        Updated DataFrame
    """
    results_dir = inventory_path.parent
    # Retry loop for concurrent access handling
    for attempt in range(max_retries):
        try:
            # Load inventory
            df = pd.read_csv(inventory_path)
            break
        except (PermissionError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (1.5 ** attempt))  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to read inventory after {max_retries} attempts: {e}")

    # If no specific recording_id, check all recordings for completion
    if recording_id is None:
        print("Checking status of all recordings...")
        for idx, row in df.iterrows():
            # Skip if output_path is NaN (not yet processed)
            if pd.isna(row['output_path']):
                df.loc[idx, 'status'] = 'pending'
                continue
            
            output_path = Path(results_dir) / Path(row['edf_path']).stem.replace('_emg', '')
            
            # Check if output files exist
            if not output_path.exists():
                df.loc[idx, 'status'] = 'pending'
                continue
            
            if check_decomposition_complete(output_path):
                df.loc[idx, 'status'] = 'completed'
                df.loc[idx, 'last_updated'] = datetime.now().isoformat()
            elif (output_path / 'error.log').exists():
                df.loc[idx, 'status'] = 'failed'
                df.loc[idx, 'last_updated'] = datetime.now().isoformat()
                # Read error message
                with open(output_path / 'error.log', 'r') as f:
                    df.loc[idx, 'error_message'] = f.read()[:200]  # First 200 chars
    
    else:
        # Update specific recording
        mask = df['recording_id'] == recording_id
        if status:
            df.loc[mask, 'status'] = status
            if status == 'completed':
                df.loc[mask, 'output_path'] = output_path
        if error_message:
            df.loc[mask, 'error_message'] = error_message
        df.loc[mask, 'last_updated'] = datetime.now().isoformat()
    
    # Save updated inventory with retry logic
    for attempt in range(max_retries):
        try:
            df.to_csv(inventory_path, index=False)
            return df
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (1.5 ** attempt))  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to write inventory after {max_retries} attempts: {e}")


def check_decomposition_complete(output_path: Path) -> bool:
    """
    Check if decomposition is complete for a recording.
    
    Args:
        output_path: Path to output directory for recording
        
    Returns:
        True if all expected output files exist
    """
    if not output_path.exists():
        return False
    
    # Check for essential output files
    required_files = [
        'predicted_sources.npz',
        'process_metadata.json'
    ]
    
    return all((output_path / f).exists() for f in required_files)


def print_status_summary(inventory_path: Path):
    """
    Print a summary of the inventory status.
    
    Args:
        inventory_path: Path to inventory CSV
    """
    df = pd.read_csv(inventory_path)
    
    print(f"Inventory Status: {inventory_path.name}")
    print(f"Total recordings: {len(df)}")
    print(f"\nStatus breakdown:")
    print(df['status'].value_counts().to_string())
    
    if (df['status'] == 'failed').any():
        print(f"\nFailed recordings:")
        failed = df[df['status'] == 'failed'][['recording_id', 'subject', 'task', 'error_message']]
        print(failed.to_string(index=False))


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description='Manage decomposition inventory for dataset-algorithm combinations')
    
    parser.add_argument('-d', '--dataset', required=True, help='Dataset name')
    parser.add_argument('-a', '--algorithm',required=True, choices=['scd', 'cbss', 'upperbound'], help='Algorithm name')
    parser.add_argument('--create', action='store_true', help='Create new inventory')
    parser.add_argument('--update', action='store_true', help='Update inventory status by checking output files')
    parser.add_argument('--status', action='store_true', help='Print status summary')
    parser.add_argument('--bids-root', type=Path, help='Override default BIDS root directory')
    parser.add_argument('--output-dir', type=Path, help='Override default output directory')
    
    args = parser.parse_args()
    
    # Determine inventory path
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(INTERIM_DIR) / f"{args.algorithm}" / args.dataset
    
    inventory_path = output_dir / f'inventory_{args.algorithm}.csv'
    
    # Execute requested action
    if args.create:
        create_inventory(
            dataset_name=args.dataset,
            algorithm=args.algorithm,
            bids_root=args.bids_root,
            output_dir=output_dir
        )
    
    if args.update:
        if not inventory_path.exists():
            print(f"Error: Inventory not found at {inventory_path}")
            print("Create it first with --create")
            return
        update_inventory_status(inventory_path)
        print(f"Updated inventory at: {inventory_path}")
    
    if args.status or args.update:
        if inventory_path.exists():
            print_status_summary(inventory_path)
        else:
            print(f"Error: Inventory not found at {inventory_path}")


if __name__ == '__main__':
    main()
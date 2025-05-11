#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from muniverse.data_generation import generate_recording

CONFIGS_DIR = '/rds/general/user/pm1222/ephemeral/muniverse/datasets/configs/neuromotion-train/'
OUTPUT_DIR = '/rds/general/user/pm1222/ephemeral/muniverse/datasets/outputs/neuromotion-train/'
CACHE_DIR = '/rds/general/user/pm1222/ephemeral/muniverse/datasets/muapcache/'

def process_configs(min_id, max_id, configs_dir: str, output_dir: str = OUTPUT_DIR, cache_dir: str = CACHE_DIR):
    """
    Process all JSON config files in the given directory.
    
    Args:
        configs_dir (str): Path to directory containing config files
        output_dir (str): Base output directory for all recordings
    """
    # Make sure all paths are absolute
    configs_dir = os.path.abspath(configs_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Ensure output directory and cache directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get all JSON files in the configs directory
    config_files = list(Path(configs_dir).glob('*.json'))
    config_files = sorted(config_files)
    
    if not config_files:
        print(f"No JSON config files found in {configs_dir}")
        return
    
    print(f"Found {len(config_files)} config files to process")
    
    # Process each config file
    for config_file in config_files[min_id: max_id]:
        print(f"\nProcessing {config_file.name}...")
        
        try:
            run_dir = generate_recording({
                'input_config': str(config_file),
                'output_dir': output_dir,
                'engine': 'singularity',
                'container': os.path.abspath('environment/muniverse-test_neuromotion.sif'),
                'cache_dir': cache_dir
            })
            print(f"Successfully generated recording in {run_dir}")
            
        except Exception as e:
            print(f"Error processing {config_file.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate recordings from multiple config files')
    parser.add_argument('-min_id', type=int, default=0, help='Minimum ID to process')
    parser.add_argument('-max_id', type=int, default=10000, help='Maximum ID to process')
    parser.add_argument('--configs_dir', default=CONFIGS_DIR, help='Directory containing JSON config files')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for recordings')
    parser.add_argument('--cache_dir', default=CACHE_DIR, help='Cache directory for MUAPs')
    
    args = parser.parse_args()
    process_configs(args.min_id, args.max_id, args.configs_dir, args.output_dir, args.cache_dir)

if __name__ == '__main__':
    main() 

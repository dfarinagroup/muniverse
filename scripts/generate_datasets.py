#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from muniverse.data_generation import generate_recording

def process_configs(configs_dir: str, output_dir: str = 'outputs'):
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
    os.makedirs(os.path.join(output_dir, 'cache'), exist_ok=True)
    
    # Get all JSON files in the configs directory
    config_files = list(Path(configs_dir).glob('*.json'))
    
    if not config_files:
        print(f"No JSON config files found in {configs_dir}")
        return
    
    print(f"Found {len(config_files)} config files to process")
    
    # Process each config file
    for config_file in config_files:
        print(f"\nProcessing {config_file.name}...")
        
        try:
            run_dir = generate_recording({
                'input_config': str(config_file),
                'output_dir': output_dir,
                'engine': 'singularity',
                'container': os.path.abspath('environment/muniverse-test_neuromotion.sif'),
                'cache_dir': os.path.join(output_dir, 'cache')
            })
            print(f"Successfully generated recording in {run_dir}")
            
        except Exception as e:
            print(f"Error processing {config_file.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate recordings from multiple config files')
    parser.add_argument('configs_dir', help='Directory containing JSON config files')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for recordings')
    
    args = parser.parse_args()
    process_configs(args.configs_dir, args.output_dir)

if __name__ == '__main__':
    main() 
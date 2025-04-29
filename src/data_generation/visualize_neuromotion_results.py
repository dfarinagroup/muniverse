#!/usr/bin/env python3
"""
Visualization script for NeuroMotion EMG simulation results.
This script creates diagnostic plots to verify the correctness of the generated signals.

Usage:
    python visualize_neuromotion_results.py <data_directory>

The data directory should contain the output files from run_neuromotion.py,
including emg.npy, spikes.npy, effort_profile.npy, etc.
"""

import os
import sys
import argparse
from plot_utils import (
    load_data, 
    preprocess_data, 
    plot_overview, 
    plot_emg_heatmap,
    plot_muap_examples,
    plot_angle_profile,
    plot_mu_recruitment
)


def plot_all(data_dir):
    """Generate all plots for a given data directory."""
    # Create plots directory
    plots_dir = os.path.join(data_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    data = load_data(data_dir)
    
    # Preprocess data to ensure correct format
    data = preprocess_data(data)
    
    # Generate all plots
    plot_overview(data, plots_dir)
    plot_emg_heatmap(data, plots_dir)
    
    if 'muaps' in data:
        plot_muap_examples(data, plots_dir)
    
    if 'angle' in data:
        plot_angle_profile(data, plots_dir)
    
    plot_mu_recruitment(data, plots_dir)
    
    print(f"All plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize NeuroMotion EMG simulation results")
    parser.add_argument("data_dir", help="Directory containing the simulation results")
    parser.add_argument("--plot", choices=["overview", "heatmap", "muaps", "angle", "recruitment", "all"], 
                        default="all", help="Specific plot to generate (default: all)")
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a valid directory")
        return 1
    
    # Create plots directory
    plots_dir = os.path.join(args.data_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load and preprocess data
    data = load_data(args.data_dir)
    data = preprocess_data(data)
    
    if not data:
        print("Error: No data files found in the specified directory")
        return 1
    
    # Generate requested plot(s)
    if args.plot == "all":
        plot_all(args.data_dir)
    elif args.plot == "overview":
        plot_overview(data, plots_dir)
    elif args.plot == "heatmap":
        plot_emg_heatmap(data, plots_dir)
    elif args.plot == "muaps":
        if 'muaps' in data:
            plot_muap_examples(data, plots_dir)
        else:
            print("Error: MUAP data not found")
    elif args.plot == "angle":
        if 'angle' in data:
            plot_angle_profile(data, plots_dir)
        else:
            print("Error: Angle profile data not found")
    elif args.plot == "recruitment":
        plot_mu_recruitment(data, plots_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
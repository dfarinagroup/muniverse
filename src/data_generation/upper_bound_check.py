import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Ensure proper path setup
sys.path.append('.')
if os.path.exists('./src'):
    sys.path.append('./src')

# Import required functions
from algorithms.decomposition_methods import upper_bound
from evaluation.evaluate import evaluate_spike_matches

def calculate_decomposition_metrics(results_path, tol=0.001, max_shift=0.1, threshold=0.3, plot_results=True, save_plots=False):
    """
    Calculate decomposition metrics for a given dataset.
    
    Args:
        results_path (str): Path to the results directory containing EMG, spikes, MUAPs, etc.
        tol (float): Tolerance for spike matching (in seconds)
        max_shift (float): Maximum time shift allowed when matching spikes (in seconds)
        threshold (float): Minimum match score required to consider a match
        plot_results (bool): Whether to plot the results
        save_plots (bool): Whether to save the plots instead of displaying them
        
    Returns:
        dict: Dictionary containing metrics and results
    """
    # Create output directory for any saved plots
    output_dir = os.path.join(results_path, 'decomposition_analysis')
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load the data
    print(f"Loading data from {results_path}")
    
    # Load EMG data
    emg_path = os.path.join(results_path, 'emg.npy')
    emg = np.load(emg_path)
    print(f"Loaded EMG data with shape: {emg.shape}")
    
    # Load config to get sampling frequency
    config_path = os.path.join(results_path, 'config_used.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    fsamp = config['RecordingConfiguration']['SamplingFrequency']
    print(f"Sampling frequency: {fsamp} Hz")
    
    # Load true spikes
    true_spikes_path = os.path.join(results_path, 'spikes.npy')
    true_spikes = np.load(true_spikes_path, allow_pickle=True)
    print(f"Loaded true spikes data with {len(true_spikes)} motor units")
    
    # Step 2: Load and process MUAPs
    # Extract muscle and DOF information from config
    muscle = config['MovementConfiguration']['TargetMuscle']
    movement_dof = config['MovementConfiguration']['MovementDOF']
    
    # Define the path to the muap cache file using the correct naming scheme
    # First check in src/data_generation/res/muap_cache
    potential_cache_paths = [
        os.path.join(os.path.dirname(results_path), "muap_cache", f"{muscle}_{movement_dof}_muaps.npy"),
        os.path.join("src", "data_generation", "res", "muap_cache", f"{muscle}_{movement_dof}_muaps.npy"),
        os.path.join("data_generation", "res", "muap_cache", f"{muscle}_{movement_dof}_muaps.npy"),
        os.path.join(results_path, "muap_cache", f"{muscle}_{movement_dof}_muaps.npy"),
        os.path.join(os.path.dirname(os.path.dirname(results_path)), "res", "muap_cache", f"{muscle}_{movement_dof}_muaps.npy")
    ]
    
    # Try each potential path until we find the file
    muap_cache_file = None
    for path in potential_cache_paths:
        if os.path.exists(path):
            muap_cache_file = path
            break
    
    if muap_cache_file is None:
        # If still not found, try to find muap_cache directory anywhere in project
        for root, dirs, files in os.walk('.'):
            if "muap_cache" in dirs:
                potential_path = os.path.join(root, "muap_cache", f"{muscle}_{movement_dof}_muaps.npy")
                if os.path.exists(potential_path):
                    muap_cache_file = potential_path
                    break
    
    # Load MUAPs from cache directory or results directory
    if muap_cache_file is not None:
        print(f"Loading MUAPs from cache: {muap_cache_file}")
        muaps_full = np.load(muap_cache_file, allow_pickle=True)
    else:
        # Last resort: check if muaps.npy exists in the results directory
        muaps_path = os.path.join(results_path, 'muaps.npy')
        if os.path.exists(muaps_path):
            print(f"Loading MUAPs from results: {muaps_path}")
            muaps_full = np.load(muaps_path, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Could not find MUAPs for {muscle}_{movement_dof} in any of the expected locations")

    
    # Extract the constant angle MUAPs
    # Get the angle profile
    angle_profile = np.load(os.path.join(results_path, 'angle_profile.npy'))
    constant_angle = angle_profile[0]
    
    # Generate angle labels
    if movement_dof == "Flexion-Extension":
        min_angle, max_angle = -65, 65
    elif movement_dof == "Radial-Ulnar-deviation":
        min_angle, max_angle = -10, 25
    else:
        min_angle, max_angle = -60, 60
    
    muap_dof_samples = muaps_full.shape[1]
    angle_labels = np.linspace(min_angle, max_angle, muap_dof_samples).astype(int)
    
    # Find the index of the angle in the MUAP library
    angle_idx = np.argmin(np.abs(angle_labels - constant_angle))
    
    # Extract MUAPs at the constant angle
    muaps = muaps_full[:, angle_idx, :, :, :]
    print(f"Extracted MUAPs for angle {angle_labels[angle_idx]}° (index {angle_idx})")
    print(f"MUAPs shape: {muaps.shape}")
    
    # Step 3: Run the decomposition
    # Reshape MUAPs from (n_mu, n_rows, n_cols, n_samples) to (n_mu, n_channels, n_samples)
    n_mu, n_rows, n_cols, n_samples = muaps.shape
    muaps_reshaped = muaps.reshape(n_mu, n_rows * n_cols, n_samples)
    
    # The decomposition method expects EMG with shape (n_channels, n_samples)
    if emg.shape[0] > emg.shape[1]:
        # If EMG has more rows than columns, it likely has shape (n_samples, n_channels)
        emg_reshaped = emg.T
    else:
        # If it already has shape (n_channels, n_samples)
        emg_reshaped = emg
    
    print(f"Running decomposition with MUAPs shape {muaps_reshaped.shape} and EMG shape {emg_reshaped.shape}")
    
    # Initialize the decomposition method
    decomposer = upper_bound(ext_fact=12, whitening_method='ZCA', cluster_method='kmeans')
    
    # Run decomposition
    sources, spikes, sil = decomposer.decompose(emg_reshaped, muaps_reshaped, fsamp)
    
    # Print initial summary
    print(f"Decomposition complete. Detected spikes in {len(spikes)} motor units")
    print(f"Average silhouette score: {np.mean(sil):.3f}")
    
    # Step 4: Convert spikes to DataFrame format for evaluation
    # Convert true spikes from array to dictionary
    true_spikes_dict = {}
    for i, unit_spikes in enumerate(true_spikes):
        if len(unit_spikes) > 0:  # Only include units with spikes
            true_spikes_dict[i] = unit_spikes
    
    # Convert spikes to dataframes
    def spikes_to_dataframe(spikes_dict, fsamp):
        data = []
        for source_id, spike_indices in spikes_dict.items():
            for spike_idx in spike_indices:
                # Convert sample index to time in seconds
                spike_time = spike_idx / fsamp
                data.append({'source_id': source_id, 'spike_time': spike_time})
        return pd.DataFrame(data)
    
    true_spikes_df = spikes_to_dataframe(true_spikes_dict, fsamp)
    discovered_spikes_df = spikes_to_dataframe(spikes, fsamp)
    
    print(f"True spikes: {len(true_spikes_df)} spikes across {len(true_spikes_dict)} units")
    print(f"Discovered spikes: {len(discovered_spikes_df)} spikes across {len(spikes)} units")
    
    # Step 5: Evaluate the decomposition performance
    # Get the total recording duration
    total_duration = sources.shape[1] / fsamp
    
    # Run the evaluation
    results = evaluate_spike_matches(
        true_spikes_df, 
        discovered_spikes_df,
        t_start=0,
        t_end=total_duration,
        tol=tol,
        max_shift=max_shift, 
        fsamp=fsamp,
        threshold=threshold
    )
    
    # Calculate summary metrics
    num_matches = len(results)
    avg_score = results['match_score'].mean() if num_matches > 0 else 0
    avg_precision = results['common_spikes'].sum() / (results['common_spikes'].sum() + results['only_df2'].sum()) if num_matches > 0 else 0
    avg_recall = results['common_spikes'].sum() / (results['common_spikes'].sum() + results['only_df1'].sum()) if num_matches > 0 else 0
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    metrics = {
        'num_motor_units': n_mu,
        'num_matched_units': num_matches,
        'match_percentage': num_matches / len(true_spikes_dict) * 100 if len(true_spikes_dict) > 0 else 0,
        'avg_match_score': avg_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1,
        'avg_silhouette': np.mean(sil),
        'min_silhouette': np.min(sil),
        'max_silhouette': np.max(sil),
        'median_silhouette': np.median(sil)
    }
    
    # Print summary metrics
    print("\nDecomposition Performance Metrics:")
    print(f"Matched {num_matches} out of {len(true_spikes_dict)} motor units ({metrics['match_percentage']:.1f}%)")
    print(f"Average match score: {avg_score:.3f}")
    print(f"Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1 Score: {avg_f1:.3f}")
    
    # Step 6: Visualize the results (if requested)
    if plot_results:
        # Plot 1: Silhouette score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sil, bins=20)
        plt.axvline(x=np.mean(sil), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(sil):.3f}')
        plt.title('Distribution of Silhouette Scores')
        plt.xlabel('Silhouette Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'silhouette_distribution.png'))
            plt.close()
        else:
            plt.show()
        
        # Plot 2: Number of spikes per motor unit
        spike_counts = [len(spike_list) for spike_list in spikes.values()]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(spike_counts)), spike_counts)
        plt.title('Number of Spikes per Motor Unit')
        plt.xlabel('Motor Unit Index')
        plt.ylabel('Number of Spikes')
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'spike_counts.png'))
            plt.close()
        else:
            plt.show()
        
        # Plot 3: Best matches comparison
        if num_matches > 0:
            num_to_plot = min(3, num_matches)
            best_matches = results.sort_values(by='match_score', ascending=False).head(num_to_plot)
            
            fig, axes = plt.subplots(num_to_plot, 1, figsize=(15, 4*num_to_plot), sharex=True)
            if num_to_plot == 1:
                axes = [axes]  # Make axes iterable if only one subplot
            
            # Focus on middle 4 seconds
            middle_start_sec = (total_duration - 4) / 2
            middle_end_sec = middle_start_sec + 4
            
            for i, (_, match) in enumerate(best_matches.iterrows()):
                true_id = int(match['source_df1'])
                est_id = int(match['source_df2'])
                score = match['match_score']
                
                # Get spike times for this unit
                true_unit_times = true_spikes_df[true_spikes_df['source_id'] == true_id]['spike_time'].values
                est_unit_times = discovered_spikes_df[discovered_spikes_df['source_id'] == est_id]['spike_time'].values
                
                # Filter to middle 4 seconds
                true_mid_times = true_unit_times[(true_unit_times >= middle_start_sec) & 
                                               (true_unit_times <= middle_end_sec)]
                est_mid_times = est_unit_times[(est_unit_times >= middle_start_sec) & 
                                             (est_unit_times <= middle_end_sec)]
                
                # Adjust to start at 0 for plotting
                true_mid_times = true_mid_times - middle_start_sec
                est_mid_times = est_mid_times - middle_start_sec
                
                # Plot
                ax = axes[i]
                
                # Create spike trains for visualization
                x_time = np.linspace(0, 4, 1000)  # 4 seconds at 1000 points
                true_train = np.zeros_like(x_time)
                est_train = np.zeros_like(x_time)
                
                for t in true_mid_times:
                    idx = int(t * 250)  # 1000 points / 4 seconds = 250 points per second
                    if 0 <= idx < len(true_train):
                        true_train[idx] = 1
                        
                for t in est_mid_times:
                    idx = int(t * 250)
                    if 0 <= idx < len(est_train):
                        est_train[idx] = 0.8  # Slightly lower to distinguish
                
                # Plot true spikes (red)
                ax.stem(x_time, true_train, linefmt='r-', markerfmt='ro', basefmt=' ', 
                       label='True Spikes')
                
                # Plot estimated spikes (blue)
                ax.stem(x_time, est_train, linefmt='b-', markerfmt='b^', basefmt=' ', 
                       label='Discovered Spikes')
                
                ax.set_title(f'True MU {true_id} vs Estimated MU {est_id} (Score: {score:.2f})')
                ax.set_ylabel('Spike Amplitude')
                ax.set_ylim(-0.2, 1.2)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
            axes[-1].set_xlabel('Time (seconds)')
            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'best_matches.png'))
                plt.close()
            else:
                plt.show()
                
            # Detailed stats on best matches
            match_details = []
            for _, match in best_matches.iterrows():
                true_id = int(match['source_df1'])
                est_id = int(match['source_df2'])
                score = match['match_score']
                common_spikes = match['common_spikes']
                only_est = match['only_df2']
                only_true = match['only_df1']
                delay = match['delay_seconds']
                
                precision = common_spikes / (common_spikes + only_est) if (common_spikes + only_est) > 0 else 0
                recall = common_spikes / (common_spikes + only_true) if (common_spikes + only_true) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                match_details.append({
                    'true_id': true_id,
                    'est_id': est_id,
                    'score': score,
                    'common_spikes': common_spikes,
                    'only_true': only_true,
                    'only_est': only_est,
                    'delay_ms': delay * 1000,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
            
            print("\nBest Matched Motor Units:")
            for i, details in enumerate(match_details):
                print(f"Match {i+1}: True MU {details['true_id']} → Estimated MU {details['est_id']}")
                print(f"  Matching score: {details['score']:.3f}")
                print(f"  Time delay: {details['delay_ms']:.2f} ms")
                print(f"  Common spikes: {details['common_spikes']}, Only in true: {details['only_true']}, " 
                      f"Only in estimated: {details['only_est']}")
                print(f"  Precision: {details['precision']:.3f}, Recall: {details['recall']:.3f}, "
                      f"F1 Score: {details['f1_score']:.3f}\n")
    
    # Step 7: Return all results and metrics
    return {
        'metrics': metrics,
        'matches': results if not results.empty else None,
        'silhouette_scores': sil,
        'spike_counts': spike_counts,
        'sources': sources,
        'estimated_spikes': spikes,
        'true_spikes': true_spikes_dict,
        'configuration': {
            'muscle': muscle,
            'movement_dof': movement_dof,
            'sampling_rate': fsamp,
            'constant_angle': angle_labels[angle_idx],
            'total_duration': total_duration,
            'evaluation_parameters': {
                'tolerance': tol,
                'max_shift': max_shift,
                'threshold': threshold
            }
        }
    }

def batch_process_decomposition(base_path, output_file='decomposition_results.csv', 
                               tol=0.001, max_shift=0.1, threshold=0.3, save_plots=True):
    """
    Process multiple datasets and compile results
    
    Args:
        base_path (str): Base directory containing multiple result folders
        output_file (str): CSV file to save combined results
        tol, max_shift, threshold: Parameters for evaluate_spike_matches
        save_plots (bool): Whether to save plots for each dataset
        
    Returns:
        pd.DataFrame: Combined results from all datasets
    """
    all_results = []
    
    # Find all result directories
    result_dirs = []
    for root, dirs, files in os.walk(base_path):
        if 'config_used.json' in files and 'emg.npy' in files and 'spikes.npy' in files:
            result_dirs.append(root)
    
    print(f"Found {len(result_dirs)} result directories to process")
    
    # Process each directory
    for i, result_dir in enumerate(result_dirs):
        print(f"\nProcessing dataset {i+1}/{len(result_dirs)}: {result_dir}")
        try:
            # Calculate decomposition metrics
            results = calculate_decomposition_metrics(
                result_dir, 
                tol=tol, 
                max_shift=max_shift, 
                threshold=threshold, 
                plot_results=save_plots, 
                save_plots=save_plots
            )
            
            # Extract metrics to add to combined results
            metrics = results['metrics']
            config = results['configuration']
            
            result_summary = {
                'dataset': os.path.basename(result_dir),
                'muscle': config['muscle'],
                'movement_dof': config['movement_dof'],
                'constant_angle': config['constant_angle'],
                'duration': config['total_duration'],
                'sampling_rate': config['sampling_rate'],
                'num_motor_units': metrics['num_motor_units'],
                'num_matched_units': metrics['num_matched_units'],
                'match_percentage': metrics['match_percentage'],
                'avg_match_score': metrics['avg_match_score'],
                'avg_precision': metrics['avg_precision'],
                'avg_recall': metrics['avg_recall'],
                'avg_f1_score': metrics['avg_f1_score'],
                'avg_silhouette': metrics['avg_silhouette'],
                'min_silhouette': metrics['min_silhouette'],
                'max_silhouette': metrics['max_silhouette'],
            }
            
            all_results.append(result_summary)
            print(f"Successfully processed {result_dir}")
            
        except Exception as e:
            print(f"Error processing {result_dir}: {str(e)}")
    
    # Combine results into DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        results_df.to_csv(os.path.join(base_path, output_file), index=False)
        print(f"\nSaved combined results to {os.path.join(base_path, output_file)}")
        
        return results_df
    else:
        print("No results were successfully processed")
        return None


# Example usage:
if __name__ == "__main__":
    # For a single dataset
    results = calculate_decomposition_metrics("./data_generation/res")
    
    # For multiple datasets
    # results_df = batch_process_decomposition("./data_generation")
    
    # Print results summary
    # print(results_df)
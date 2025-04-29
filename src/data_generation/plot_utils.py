#!/usr/bin/env python3
"""
Utility functions for visualizing NeuroMotion EMG simulation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_data(data_dir):
    """Load all relevant data files from the specified directory."""
    files = {
        'emg': os.path.join(data_dir, 'emg.npy'),
        'spikes': os.path.join(data_dir, 'spikes.npy'),
        'effort': os.path.join(data_dir, 'effort_profile.npy'),
        'angle': os.path.join(data_dir, 'angle_profile.npy'),
        'muaps': os.path.join(data_dir, 'muaps.npy'),
        'metadata': os.path.join(data_dir, 'metadata.json'),
        'config': os.path.join(data_dir, 'config_used.json')
    }
    
    data = {}
    for key, file_path in files.items():
        if os.path.exists(file_path):
            if file_path.endswith('.npy'):
                try:
                    # First try loading with default settings
                    data[key] = np.load(file_path)
                except ValueError:
                    # If that fails, try with allow_pickle=True
                    data[key] = np.load(file_path, allow_pickle=True)
                print(f"Loaded {key} from {file_path}")
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    import json
                    data[key] = json.load(f)
                print(f"Loaded {key} from {file_path}")
        else:
            print(f"Warning: {file_path} not found")
    
    return data


def reshape_emg(emg, n_rows, n_cols):
    """
    Reshape EMG data from 2D (time_samples, n_channels) to 3D (n_rows, n_cols, time_samples).
    
    Args:
        emg (numpy.ndarray): EMG data in shape (time_samples, n_channels)
        n_rows (int): Number of rows in the electrode array
        n_cols (int): Number of columns in the electrode array
    
    Returns:
        numpy.ndarray: Reshaped EMG data in shape (n_rows, n_cols, time_samples)
    """
    time_samples, n_channels = emg.shape
    
    # Ensure the number of channels matches rows*cols
    if n_channels != n_rows * n_cols:
        raise ValueError(f"Channel count {n_channels} doesn't match n_rows*n_cols ({n_rows}*{n_cols}={n_rows*n_cols})")
    
    # Reshape to 3D format
    emg_reshaped = np.zeros((n_rows, n_cols, time_samples))
    
    for t in range(time_samples):
        # Reshape the channel data at time t into the electrode array
        channel_data = emg[t, :].reshape(n_rows, n_cols)
        emg_reshaped[:, :, t] = channel_data
    
    return emg_reshaped


def preprocess_data(data):
    """
    Preprocess loaded data to ensure it's in the correct format for visualization.
    
    Args:
        data (dict): Dictionary containing the loaded data
    
    Returns:
        dict: Preprocessed data
    """
    # If EMG data is available, check and reshape if needed
    if 'emg' in data:
        emg = data['emg']
        
        # Get electrode array dimensions from metadata
        n_rows = 32  # Default
        n_cols = 10  # Default
        
        if 'metadata' in data and 'simulation_info' in data['metadata']:
            if 'electrode_array' in data['metadata']['simulation_info']:
                electrode_info = data['metadata']['simulation_info']['electrode_array']
                n_rows = electrode_info.get('rows', 32)
                n_cols = electrode_info.get('columns', 10)
        
        # Detect the format based on dimensionality
        if emg.ndim == 2:
            # Format: (time_samples, n_channels)
            time_samples, n_channels = emg.shape
            
            # If channels match n_rows*n_cols, reshape to 3D array
            if n_channels == n_rows * n_cols:
                print(f"Reshaping EMG from (time_samples, n_channels) to (n_rows, n_cols, time_samples)")
                try:
                    # Attempt direct reshape if the data is arranged properly
                    data['emg'] = emg.reshape(time_samples, n_rows, n_cols).transpose(1, 2, 0)
                except Exception:
                    # If direct reshape fails, try the element-by-element method
                    data['emg'] = reshape_emg(emg, n_rows, n_cols)
            else:
                print(f"Warning: Channel count {n_channels} doesn't match n_rows*n_cols ({n_rows}*{n_cols}={n_rows*n_cols})")
        
        elif emg.ndim == 3:
            # Check if format is (time_samples, n_rows, n_cols)
            # If so, transpose to (n_rows, n_cols, time_samples)
            if emg.shape[1] == n_rows and emg.shape[2] == n_cols:
                data['emg'] = emg.transpose(1, 2, 0)
    return data


def is_sparse_spike_format(spikes):
    """Check if spikes are in sparse format (lists of indices) or dense format (binary arrays)."""
    if not isinstance(spikes, (list, np.ndarray)):
        return False
    
    # Check the first few elements
    for i in range(min(5, len(spikes))):
        if isinstance(spikes[i], list) and all(isinstance(x, (int, np.integer)) for x in spikes[i][:10]):
            return True
    
    return False


def plot_overview(data, output_dir):
    """
    Create an overview plot showing EMG signals, effort profile, and active motor units.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'emg' not in data or 'effort' not in data or 'spikes' not in data:
        print("Error: Missing required data for overview plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 3, figure=fig)
    
    # Get metadata for plotting
    fs = 2048  # Default sampling frequency
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        fs = data['metadata']['simulation_info'].get('fs', 2048)
    
    # Time vectors
    emg = data['emg']
    if emg.ndim == 3:
        # EMG format is (n_rows, n_cols, time_samples)
        n_rows, n_cols, time_samples = emg.shape
        time_emg = np.arange(time_samples) / fs
    else:
        # EMG format is (time_samples, n_channels)
        time_samples, _ = emg.shape
        time_emg = np.arange(time_samples) / fs
    
    time_effort = np.arange(len(data['effort'])) / fs
    
    # Plot EMG signals from multiple channels
    ax1 = fig.add_subplot(gs[0, :])
    
    if emg.ndim == 3:
        # EMG in (n_rows, n_cols, time_samples) format
        n_rows, n_cols, _ = emg.shape
        center_row, center_col = n_rows // 2, n_cols // 2
        
        # Plot a few channels from the center of the array
        channels_to_plot = [
            (center_row, center_col),
            (center_row - min(3, center_row), center_col),
            (center_row + min(3, n_rows - center_row - 1), center_col),
            (center_row, center_col - min(2, center_col)),
            (center_row, center_col + min(2, n_cols - center_col - 1))
        ]
        
        for i, (row, col) in enumerate(channels_to_plot):
            if 0 <= row < n_rows and 0 <= col < n_cols:
                ax1.plot(time_emg, emg[row, col, :] + i*0.5, label=f"Channel ({row},{col})")
    else:
        # EMG in (time_samples, n_channels) format
        n_channels = emg.shape[1]
        n_to_plot = min(5, n_channels)
        
        # Plot a few channels evenly spaced in the array
        channel_indices = np.linspace(0, n_channels-1, n_to_plot, dtype=int)
        
        for i, ch_idx in enumerate(channel_indices):
            ax1.plot(time_emg, emg[:, ch_idx] + i*0.5, label=f"Channel {ch_idx}")
    
    ax1.set_title("EMG Signals from Selected Channels")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot effort profile
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_effort, data['effort'], 'r-', linewidth=2)
    ax2.set_title("Effort Profile")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Effort Level")
    ax2.grid(True, alpha=0.3)
    
    # Check if spikes are in sparse format (lists of spike times) or dense format (binary arrays)
    sparse_format = is_sparse_spike_format(data['spikes'])
    spikes = data['spikes']
    num_mus = len(spikes)
    
    # Plot number of active motor units over time
    ax3 = fig.add_subplot(gs[2, :])
    
    # Calculate active MUs using appropriate method for the spike format
    window_size = int(fs * 0.1)  # 100ms window
    if sparse_format:
        # For sparse format (lists of spike indices)
        active_mus = np.zeros(len(data['effort']) - window_size + 1)
        for t in range(len(active_mus)):
            active_count = 0
            window_start = t
            window_end = t + window_size
            
            for mu_idx in range(num_mus):
                # Check if any spike falls within this window
                if len(spikes[mu_idx]) > 0:
                    if any(window_start <= spike < window_end for spike in spikes[mu_idx] if spike < len(data['effort'])):
                        active_count += 1
            
            active_mus[t] = active_count
    else:
        # For dense format (binary arrays)
        active_mus = np.zeros(len(data['effort']) - window_size + 1)
        for j in range(len(active_mus)):
            active_count = 0
            for i in range(num_mus):
                if np.sum(spikes[i][j:j+window_size]) > 0:
                    active_count += 1
            active_mus[j] = active_count
    
    time_active = np.arange(len(active_mus)) / fs
    ax3.plot(time_active, active_mus, 'b-', linewidth=2)
    ax3.set_title("Number of Active Motor Units")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Active MUs")
    ax3.set_ylim(0, num_mus * 1.1)  # Give some headroom
    ax3.grid(True, alpha=0.3)
    
    # Plot spike trains (raster plot)
    ax4 = fig.add_subplot(gs[3, :])
    
    # Plot subset of spike trains as raster plot
    step = max(1, num_mus // 20)  # Show at most ~20 spike trains
    for i, mu_idx in enumerate(range(0, num_mus, step)):
        if mu_idx < num_mus:
            if sparse_format:
                # For sparse format, spike times are already available
                spike_times = np.array(spikes[mu_idx]) / fs
                valid_spikes = spike_times[spike_times < max(time_emg)]
                if len(valid_spikes) > 0:
                    ax4.plot(valid_spikes, np.ones_like(valid_spikes) * i, '|', markersize=3)
            else:
                # For dense format, convert to spike times
                spike_times = np.where(spikes[mu_idx])[0] / fs
                ax4.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=3)
    
    ax4.set_title("Spike Raster Plot (subset of MUs)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Motor Unit Index")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'overview_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot to {output_path}")
    plt.close(fig)


def plot_emg_heatmap(data, output_dir):
    """
    Create a heatmap of the EMG signal across the electrode array.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'emg' not in data:
        print("Error: EMG data missing for heatmap plot")
        return
    
    emg = data['emg']
    
    # Get electrode dimensions
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        if 'electrode_array' in data['metadata']['simulation_info']:
            electrode_info = data['metadata']['simulation_info']['electrode_array']
            n_rows = electrode_info.get('rows', 8)
            n_cols = electrode_info.get('columns', 8)
        else:
            n_rows = 8
            n_cols = 8
    else:
        n_rows = 8
        n_cols = 8
    
    # Handle different EMG formats
    if emg.ndim == 3:
        # EMG in (n_rows, n_cols, time_samples) format
        n_rows, n_cols, n_samples = emg.shape
    else:
        # EMG in (time_samples, n_channels) format
        n_samples, n_channels = emg.shape
        
        # Reshape if needed and possible
        if n_channels == n_rows * n_cols:
            # Reshape into 3D array for visualization
            emg_3d = np.zeros((n_rows, n_cols, n_samples))
            for t in range(n_samples):
                emg_3d[:, :, t] = emg[t, :].reshape(n_rows, n_cols)
            emg = emg_3d
        else:
            print(f"Warning: Cannot create heatmap - channel count {n_channels} doesn't match n_rows*n_cols ({n_rows}*{n_cols})")
            return
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get interesting time points (near peak effort)
    if 'effort' in data:
        peak_idx = np.argmax(data['effort'])
    else:
        peak_idx = n_samples // 2
    
    time_points = [
        peak_idx - min(1000, peak_idx),  # Before peak
        peak_idx,                        # At peak
        min(peak_idx + 1000, n_samples - 1),  # After peak
        max(0, n_samples - 2000)         # Near end
    ]
    
    # Ensure time points are valid
    time_points = [max(0, min(t, n_samples-1)) for t in time_points]
    
    # Compute RMS of EMG in windows around each time point
    window_size = min(500, n_samples // 10)  # samples
    heatmaps = []
    
    for t in time_points:
        start = max(0, t - window_size//2)
        end = min(n_samples, t + window_size//2)
        window_data = emg[:, :, start:end]
        rms = np.sqrt(np.mean(window_data**2, axis=2))
        heatmaps.append(rms)
    
    titles = ["Before Peak", "At Peak", "After Peak", "Near End"]
    
    # Plot heatmaps
    for i, (ax, heatmap, title) in enumerate(zip(axs.flat, heatmaps, titles)):
        im = ax.imshow(heatmap, cmap='jet', interpolation='nearest', aspect='auto')
        ax.set_title(f"EMG RMS Heatmap - {title}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, label="RMS Amplitude")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'emg_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved EMG heatmap to {output_path}")
    plt.close(fig)


def plot_muap_examples(data, output_dir):
    """
    Plot examples of MUAPs for different motor units.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'muaps' not in data:
        print("Error: MUAP data missing for MUAP example plot")
        return
    
    muaps = data['muaps']
    
    # Check if muaps have the expected shape
    if muaps.ndim != 5:
        print(f"Error: Unexpected MUAP shape: {muaps.shape}. Expected 5 dimensions.")
        return
    
    num_mus, steps, n_rows, n_cols, time_length = muaps.shape
    
    # Select a few MUs to plot
    mus_to_plot = [
        0,              # First MU (typically smallest)
        num_mus // 4,
        num_mus // 2,
        3 * num_mus // 4,
        min(num_mus - 1, num_mus - 1)     # Last MU (typically largest)
    ]
    
    # Select center of electrode array for visualization
    center_row, center_col = n_rows // 2, n_cols // 2
    
    # Create grid of subplots
    fig, axs = plt.subplots(len(mus_to_plot), 3, figsize=(15, 12))
    
    for i, mu_idx in enumerate(mus_to_plot):
        if mu_idx < num_mus:
            # Plot MUAP for different positions (steps): beginning, middle, end
            step_indices = [0, steps//2, steps-1]
            
            for j, step_idx in enumerate(step_indices):
                ax = axs[i, j]
                try:
                    muap = muaps[mu_idx, step_idx, center_row, center_col, :]
                    ax.plot(muap)
                    
                    if j == 0:
                        ax.set_ylabel(f"MU {mu_idx}")
                    
                    if i == 0:
                        if j == 0:
                            ax.set_title("Beginning Position")
                        elif j == 1:
                            ax.set_title("Middle Position")
                        else:
                            ax.set_title("End Position")
                    
                    if i == len(mus_to_plot) - 1:
                        ax.set_xlabel("Samples")
                except Exception as e:
                    print(f"Error plotting MUAP for MU {mu_idx}, step {step_idx}: {e}")
    
    plt.suptitle("Example MUAPs at Different Joint Positions (Center Electrode)", fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'muap_examples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MUAP examples to {output_path}")
    plt.close(fig)


def plot_angle_profile(data, output_dir):
    """
    Plot angle profile and corresponding effort level.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'angle' not in data:
        print("Error: Angle data missing for angle profile plot")
        return
    
    # Get metadata for plotting
    fs = 2048  # Default sampling frequency
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        fs = data['metadata']['simulation_info'].get('fs', 2048)
    
    angle_profile = data['angle']
    time_angle = np.arange(len(angle_profile)) / fs
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot angle profile
    ax1.plot(time_angle, angle_profile, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Plot effort on secondary axis if available
    if 'effort' in data:
        effort_profile = data['effort']
        
        # Make sure the effort profile is the same length as the angle profile
        if len(effort_profile) != len(angle_profile):
            # Resample effort to match angle length
            from scipy.interpolate import interp1d
            time_effort = np.arange(len(effort_profile)) / fs
            effort_interp = interp1d(time_effort, effort_profile, bounds_error=False, fill_value='extrapolate')
            effort_profile = effort_interp(time_angle)
        
        ax2 = ax1.twinx()
        ax2.plot(time_angle, effort_profile, 'r--', linewidth=2)
        ax2.set_ylabel('Effort Level', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Angle Profile and Effort Level')
    else:
        plt.title('Angle Profile')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'angle_profile.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved angle profile to {output_path}")
    plt.close(fig)


def plot_mu_recruitment(data, output_dir):
    """
    Plot motor unit recruitment patterns against effort level.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'spikes' not in data or 'effort' not in data:
        print("Error: Missing spike or effort data for recruitment plot")
        return
    
    spikes = data['spikes']
    effort = data['effort']
    
    # Get sampling frequency
    fs = 2048  # Default sampling frequency
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        fs = data['metadata']['simulation_info'].get('fs', 2048)
    
    # Check if spikes are in sparse format (lists of spike times) or dense format (binary arrays)
    sparse_format = is_sparse_spike_format(spikes)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1, 2]})
    
    # Plot 1: Effort profile
    time_effort = np.arange(len(effort)) / fs
    axs[0].plot(time_effort, effort, 'r-', linewidth=2)
    axs[0].set_title("Effort Profile")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Effort Level")
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Number of active motor units over time
    num_mus = len(spikes)
    window_size = int(fs * 0.1)  # 100ms window
    window_size = min(window_size, len(effort) // 10)  # Make sure window isn't too large
    
    if sparse_format:
        # For sparse format (lists of spike indices)
        active_mus = np.zeros(len(effort) - window_size + 1)
        for t in range(len(active_mus)):
            active_count = 0
            window_start = t
            window_end = t + window_size
            
            for mu_idx in range(num_mus):
                # Check if any spike falls within this window
                if len(spikes[mu_idx]) > 0:
                    if any(window_start <= spike < window_end for spike in spikes[mu_idx] if spike < len(effort)):
                        active_count += 1
            
            active_mus[t] = active_count
    else:
        # For dense format (binary arrays)
        active_mus = np.zeros(len(effort) - window_size + 1)
        for j in range(len(active_mus)):
            active_count = 0
            for i in range(num_mus):
                if np.sum(spikes[i][j:j+window_size]) > 0:
                    active_count += 1
            active_mus[j] = active_count
    
    time_active = np.arange(len(active_mus)) / fs
    axs[1].plot(time_active, active_mus, 'b-', linewidth=2)
    axs[1].set_title("Number of Active Motor Units")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Active MUs")
    axs[1].set_ylim(0, num_mus * 1.1)  # Give some headroom
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Detailed recruitment onset visualization
    # Calculate first spike time for each MU
    first_spike_times = np.full(num_mus, np.nan)
    
    if sparse_format:
        # For sparse format (lists of spike indices)
        for i in range(num_mus):
            if len(spikes[i]) > 0:
                first_spike_times[i] = spikes[i][0] / fs
    else:
        # For dense format (binary arrays)
        for i in range(num_mus):
            spike_indices = np.where(spikes[i])[0]
            if len(spike_indices) > 0:
                first_spike_times[i] = spike_indices[0] / fs
    
    # Sort MUs by recruitment time
    recruitment_order = np.argsort(first_spike_times)
    sorted_times = first_spike_times[recruitment_order]
    
    # Filter out NaNs (MUs that never fired)
    valid_indices = ~np.isnan(sorted_times)
    sorted_times = sorted_times[valid_indices]
    sorted_mus = recruitment_order[valid_indices]
    
    if len(sorted_times) > 0:
        # Plot recruitment times as scatter plot
        axs[2].scatter(sorted_times, np.arange(len(sorted_times)), marker='|', s=100, color='b')
        
        # Add effort profile as background reference
        ax_effort = axs[2].twinx()
        ax_effort.plot(time_effort, effort, 'r-', alpha=0.3)
        ax_effort.set_ylabel("Effort Level", color='r')
        ax_effort.tick_params(axis='y', labelcolor='r')
        
        axs[2].set_title("Motor Unit Recruitment Order")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("MU Index (sorted by recruitment time)")
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "No active motor units found", ha='center', va='center')
    
    # Create additional figure for recruitment threshold analysis
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate recruitment threshold (effort level at first spike)
    recruitment_thresholds = np.full(num_mus, np.nan)
    
    if sparse_format:
        # For sparse format (lists of spike indices)
        for i in range(num_mus):
            if len(spikes[i]) > 0:
                first_spike_idx = spikes[i][0]
                if first_spike_idx < len(effort):
                    recruitment_thresholds[i] = effort[first_spike_idx]
    else:
        # For dense format (binary arrays)
        for i in range(num_mus):
            spike_indices = np.where(spikes[i])[0]
            if len(spike_indices) > 0:
                first_spike_idx = spike_indices[0]
                if first_spike_idx < len(effort):
                    recruitment_thresholds[i] = effort[first_spike_idx]
    
    # Filter out NaNs
    valid_indices = ~np.isnan(recruitment_thresholds)
    thresholds = recruitment_thresholds[valid_indices]
    mu_indices = np.arange(num_mus)[valid_indices]
    
    if len(thresholds) > 0:
        # Plot recruitment thresholds
        ax.plot(mu_indices, thresholds, 'o-')
        ax.set_title("Motor Unit Recruitment Thresholds")
        ax.set_xlabel("Motor Unit Index")
        ax.set_ylabel("Effort Level at Recruitment")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No motor unit activation detected", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plots
    output_path1 = os.path.join(output_dir, 'mu_recruitment.png')
    fig.savefig(output_path1, dpi=300, bbox_inches='tight')
    
    output_path2 = os.path.join(output_dir, 'recruitment_thresholds.png')
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    
    print(f"Saved MU recruitment plots to {output_path1} and {output_path2}")
    plt.close(fig)
    plt.close(fig2)
from typing import List, Tuple

import numpy as np


def add_noise_to_emg(emg: np.ndarray, config: dict) -> np.ndarray:
    """
    Add noise to EMG signal if specified in config.

    Args:
        emg: EMG signal array with shape (n_rows, n_cols, T)
        config: Configuration dictionary

    Returns:
        EMG signal with noise added
    """
    recording_cfg = config["RecordingConfiguration"]
    noise_seed = recording_cfg.get("NoiseSeed")
    noise_db = recording_cfg.get("NoiseLeveldb")

    if noise_seed is None or noise_db is None:
        return emg

    rng = np.random.RandomState(noise_seed)
    std_noise = emg.std() * 10 ** (-noise_db / 20)
    return emg + rng.normal(0, std_noise, emg.shape)


def post_process_emg(config: dict, emg: np.ndarray) -> np.ndarray:
    """
    Apply post-processing (noise, electrode selection).

    Args:
        config: Configuration dictionary
        emg: EMG signal array with shape (n_rows, n_cols, T)

    Returns:
        Post-processed EMG signal array
    """
    emg = add_noise_to_emg(emg, config)
    emg, _ = select_optimal_electrodes(emg, config)
    return emg


def select_optimal_electrodes(emg: np.ndarray, config: dict) -> Tuple[np.ndarray, List]:
    """
    Select optimal electrode subset based on column RMS energy.

    Args:
        emg: EMG signal array with shape (n_rows, n_cols, T)
        config: Configuration dictionary

    Returns:
        Tuple of (selected_emg, selected_column_indices)
    """
    electrode_cfg = config["RecordingConfiguration"]["ElectrodeConfiguration"]
    n_cols = electrode_cfg["NCols"]
    desired_cols = electrode_cfg.get("DesiredNCols", n_cols)

    if desired_cols >= n_cols:
        return emg, list(range(n_cols))

    # RMS per column, averaged across rows and time
    rms_per_col = np.sqrt(np.mean(np.square(emg), axis=(0, 2)))
    center_col = int(np.argmax(rms_per_col))

    # Wrap around the grid to select the desired number of columns
    half_width = desired_cols // 2
    selected_cols = [(center_col - half_width + i) % n_cols for i in range(desired_cols)]

    return emg[:, selected_cols, :], selected_cols

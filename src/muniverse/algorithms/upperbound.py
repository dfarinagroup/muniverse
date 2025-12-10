import numpy as np
from scipy.stats import skew
from .core import extension, whitening, est_spike_times, remove_bad_sources


class UpperBound:
    """
    Class for computing an upper bound of convolutive blind source
    separation (cBSS) based motor neuron identification making use
    of a known ground-truth.
    """

    def __init__(self, config=None, **kwargs):
        # Default parameters
        self.ext_fact = 12
        self.whitening_method = "ZCA"
        self.whitening_reg = "auto"
        self.cluster_method = "kmeans"
        self.sil_th = 0.9
        self.min_num_spikes = 10

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        # Merge with directly passed keyword arguments (overwrites config)
        params = {**config_dict, **kwargs}

        valid_keys = self.__dict__.keys()

        # Assign all parameters as attributes
        for key, value in params.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

    def set_param(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")

    def decompose(self, sig, muaps, fsamp, return_all_sources=False):
        """
        Estimate the spike response of motor neurons given the
        motor unit action potentials (MUAPs)

        Args:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            muaps (ndarray): MUAPs (mu_index x n_channels x duration)
            fsamp (float): Sampling rate of the data (unit: Hz)

        Returns:
            sources (np.ndarray): Estimated sources (n_mu x n_samples)
            spikes (dict): Spiking instances of the motor neurons
            sil (np.ndarray): Source quality metric
            mu_filters (np.ndarray): Motor unit filters
        """

        n_mu = muaps.shape[0]
        sources = np.zeros((n_mu, sig.shape[1]))
        spikes = {i: [] for i in range(n_mu)}
        sil = np.zeros(n_mu)
        # Initialize mu_filters array
        white_dim = sig.shape[0] * self.ext_fact  # Dimension after extension
        mu_filters = np.zeros((white_dim, n_mu))

        # Extend signals and subtract the mean
        ext_sig = extension(sig, self.ext_fact)

        ext_mean = np.mean(ext_sig, axis=1, keepdims=True)
        ext_sig -= ext_mean

        # Remove the edges from the exteneded signal
        ext_sig[:, : self.ext_fact * 2] = 0
        ext_sig[:, -self.ext_fact * 2 :] = 0

        # Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method)
        # Loop over each MU
        for i in np.arange(n_mu):
            # Get the optimal MU filter
            w = self.muap_to_filter(muaps[i, :, :], ext_mean, Z)
            # Estimate source
            sources[i, :] = w.T @ white_sig
            # Make sure the peaks are in positive direction
            sources[i, :] = np.sign(skew(sources[i, :])) * sources[i, :]
            spikes[i], sil[i] = est_spike_times(
                sources[i, :], fsamp, cluster=self.cluster_method
            )
            # Store the filter
            mu_filters[:, i] = w

        # Remove bad sources
        if not return_all_sources:
            sources, spikes, sil, mu_filters = remove_bad_sources(
                sources,
                spikes,
                sil,
                mu_filters,
                threshold=self.sil_th,
                min_num_spikes=self.min_num_spikes,
            )
        return sources, spikes, sil, mu_filters

    def muap_to_filter(self, muap, ext_mean, Z):
        """
        Get the optimal motor unit filter from the ground truth MUAP.
        Therefore, the MUAP is extended and whitened. The optimal motor unit
        filter corresponds to the column of the extended and whitened MUAP
        that has the highest norm.

        Args:
            MUAP (ndarray): Multichannel MUAP (n_channels x duration)
            Z (ndarray): Whitening matrix

        Returns:
            w (ndarray): Normalized motor unit filter
        """

        # Extend the MUAP
        ext_muap = extension(muap, self.ext_fact)
        # ext_muap -= ext_mean

        # Whiten the MUAP
        white_muap = Z @ ext_muap

        # Find the column with the largest L2 norm and return it as MUAP filter
        col_norms = np.linalg.norm(white_muap, axis=0)
        col_norms[: self.ext_fact] = 0
        w = white_muap[:, np.argmax(col_norms)]

        # Normalize w
        w = w / np.linalg.norm(w)

        return w

    def _write_pipeline_sidecar(self):
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass


def process_neuromotion_muaps(muap_cache, simulation_config):
    """
    Load and prepare MUAPs for decomposition for a given simulated recording.
    I.e., pick MUAPs for target angle --> select subset of electrodes used during simulation

    Args:
        muap_cache: path to MUAP cache file
        simulation_config: simulation config dict

    Returns:
        processed_muaps: Processed MUAPs ready for decomposition
    """    
    # Extract configuration from the simulation config
    config = simulation_config.get('InputData', {}).get('Configuration', {})
    movement_config = config.get('MovementConfiguration', {})
    movement_dof = movement_config.get('MovementDOF')
    
    # Generate angle labels
    if movement_dof == "Flexion-Extension":
        min_angle, max_angle = -65, 65
    elif movement_dof == "Radial-Ulnar-deviation":
        min_angle, max_angle = -10, 25

    constant_angle = movement_config.get("MovementProfileParameters", {}).get('TargetAngle')
    muap_dof_samples = muap_cache.shape[1]
    angle_labels = np.linspace(min_angle, max_angle, muap_dof_samples).astype(int)

    # Find the index of the angle in the MUAP cache
    angle_idx = np.argmin(np.abs(angle_labels - constant_angle))
    muaps = muap_cache[:, angle_idx, :, :, :]

    # Reshape MUAPs from (n_mu, n_rows, n_cols, n_samples) to (n_mu, n_channels, n_samples)
    n_mu, n_rows, n_cols, n_samples = muaps.shape
    
    # Check if we need to use subset of electrodes based on simulation config
    selected_indices = None
    electrode_config = config.get('RecordingConfiguration', {}).get('ElectrodeConfiguration', {})
    desired_n_cols = electrode_config.get('DesiredNCols')
    
    if desired_n_cols < n_cols:
        # Biomime grid wraps around -- use modulo to handle wrapping
        # Calculate how many columns to take on each side of the center column
        # TODO: Move the center column from OutputData.Metadata to ElectrodeConfiguration
        center_column = simulation_config['OutputData']['Metadata'].get('CenterColumn')
        half_width = desired_n_cols // 2
        selected_columns = [(center_column - half_width + i) % n_cols for i in range(desired_n_cols)]
        selected_indices = []
        for col in selected_columns:
            selected_indices.extend([col * n_rows + row for row in range(n_rows)])
    
    # Reshape the MUAPs
    processed_muaps = muaps.reshape(n_mu, n_rows * n_cols, n_samples)
    if selected_indices is not None:
        processed_muaps = processed_muaps[:, selected_indices, :]
        print(f"Extracted MUAPs for angle {constant_angle}, and subset of {len(selected_indices)} electrodes")
    else:
        print(f"Extracted MUAPs for angle {constant_angle}, using all {n_rows * n_cols} electrodes")
    
    return processed_muaps


def process_hybrid_tibialis_muaps(muap_cache, subject_config):
    """
    Load and prepare MUAPs for decomposition for a given hybrid tibialis recording.
    """
    return muap_cache[subject_config['simulation_info']['selected_indices']]
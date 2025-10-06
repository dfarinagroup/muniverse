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

    def decompose(self, sig, muaps, fsamp):
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

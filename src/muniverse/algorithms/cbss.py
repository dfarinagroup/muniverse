import numpy as np
from scipy.stats import skew
from .core import bandpass_signals, notch_signals, extension, whitening, est_spike_times, remove_duplicates, remove_bad_sources, gram_schmidt, peel_off

class CBSS:
    """
    Class for performing convolutive blind source separation to identify the
    spiking activity of motor neurons using the fastICA algorithm.

    """

    def __init__(self, config=None, **kwargs):

        # Default parameters
        self.bandpass = [20, 500]
        self.bandpass_order = 2
        self.notch_frequency = 50
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1
        self.ext_fact = 12
        self.whitening_method = "ZCA"
        self.whitening_reg = "auto"
        self.ica_n_iter = 100
        self.opt_initalization = "random"
        self.opt_function_exp = 3
        self.opt_max_iter = 100
        self.opt_tol = 1e-4
        self.source_deflation = "gram-schmidt"
        self.peel_off = True
        self.cluster_method = "kmeans"
        self.random_seed = 1909
        self.refinement_loop = True
        self.sil_th = 0.9
        self.cov_th = 0.35
        self.min_num_spikes = 10
        self.match_th = 0.3
        self.match_max_shift = 0.1
        self.match_tol = 0.001

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

    def decompose(self, sig, fsamp):
        """
        Run simple decomposition

        Args:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            fsamp (float): Sampling frequency in Hz

        Returns:
            sources (ndarray): Estimated spike responses (n_mu x n_samples)
            spikes (dict): Sample indices of motor neuron discharges
            sil (ndarray): Pseudo-silhouette scores of the estimated sources
            mu_filters (ndarray): Optimized motor unit filters
        """

        # Initalize random number generator
        rng = np.random.seed(self.random_seed)

        # Bandpass filter signals
        if self.bandpass is not None:
            sig = bandpass_signals(
                sig,
                fsamp,
                high_pass=self.bandpass[0],
                low_pass=self.bandpass[1],
                order=self.bandpass_order,
            )

        # Notch filter signals
        if self.notch_frequency is not None:
            sig = notch_signals(
                sig,
                fsamp,
                nfreq=self.notch_frequency,
                dfreq=self.notch_width,
                order=self.notch_order,
                n_harmonics=self.notch_n_harmonics,
            )

        # Extend signals and subtract the mean and cut the edges
        ext_sig = extension(sig, self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

        # Remove the edges from the exteneded signal
        ext_sig[:, : self.ext_fact * 2] = 0
        ext_sig[:, -self.ext_fact * 2 :] = 0

        # Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method)

        # Initalize the output variables
        sources = np.zeros((self.ica_n_iter, sig.shape[1]))
        spikes = {i: [] for i in range(self.ica_n_iter)}
        sil = np.zeros(self.ica_n_iter)
        mu_filters = np.zeros((white_sig.shape[0], self.ica_n_iter))

        if self.opt_initalization == "activity_idx":
            act_idx_histoty = np.array([])

        # Loop over each MU
        for i in range(self.ica_n_iter):
            # Initalize
            if self.opt_initalization == "random":
                w = np.random.randn(white_sig.shape[0])
            elif self.opt_initalization == "activity_idx":
                col_norms = np.linalg.norm(white_sig, axis=0)
                col_norms[act_idx_histoty.astype(int)] = 0
                best_idx = np.argmax(col_norms)
                w = white_sig[:, best_idx]
                act_idx_histoty = np.append(act_idx_histoty, best_idx)
            else:
                ValueError("The specified initalization method is not implemented")

            # fastICA fixedpoint optimization
            w, k = self.my_fixed_point_alg(w, white_sig, mu_filters)

            # Predict source and estimate the source quality
            sources[i, :] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(
                sources[i, :], fsamp, cluster=self.cluster_method
            )
            if len(spikes[i]) > 2:
                isi = np.diff(spikes[i] / fsamp)
                cov = np.std(isi) / np.mean(isi)
            else:
                cov = np.inf

            # Refinement loop
            if len(spikes[i]) > 10 and self.refinement_loop:
                w, _, cov = self.mimimize_covisi(w, white_sig, cov, fsamp)
                sources[i, :] = w.T @ white_sig
                spikes[i], sil[i] = est_spike_times(
                    sources[i, :], fsamp, cluster=self.cluster_method
                )

            # Save the optimized MU filter
            mu_filters[:, i] = w

            # Peel-off the detected source
            if self.peel_off and sil[i] > self.sil_th and cov < self.cov_th:
                white_sig, _, _ = peel_off(white_sig, spikes[i], win=0.025, fsamp=fsamp)

        # Remove duplicates
        sources, spikes, sil, mu_filters = remove_duplicates(
            sources,
            spikes,
            sil,
            mu_filters,
            fsamp,
            max_shift=self.match_max_shift,
            tol=self.match_tol,
            threshold=self.match_th,
        )

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

    def my_fixed_point_alg(self, w, X, B):
        """
        Fixed-point optimization to maximize sparseness of a source signal.

        Args:
            w (np.ndarray): Initial weight vector (n_channels,)
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            B (np.ndarray): Current separation matrix (n_components x n_channels)

        Returns:
            w (np.ndarray): Optimized weight vector
            k (int): Number of iterations taken
        """

        # Define contrast function and its derivative
        # Use g(x)=x*(x**2+epsilon)**((a-1)/2) as smooth approximation of g(x) = sign(x) * abs(x)**a
        epsilon = 1e-3
        a = self.opt_function_exp
        g = lambda x: (epsilon + x**2) ** ((a - 3) / 2) * (a * x**2 + epsilon)
        gp = (
            lambda x: (a - 1)
            * x
            * (epsilon + x**2) ** ((a - 5) / 2)
            * (a * x**2 + 3 * epsilon)
        )
        # g = lambda x: x**2
        # gp = lambda x: 2*x

        TOL = self.opt_tol
        delta = np.ones(self.opt_max_iter)
        k = 0

        while delta[k] > TOL and k < self.opt_max_iter - 1:
            w_last = w.copy()

            wTX = w.T @ X  # shape: (n_samples,)
            A = np.mean(gp(wTX))
            w = np.mean(X * g(wTX), axis=1) - A * w  # shape: (n_channels,)

            # Orthogonalization step
            if self.source_deflation == "projection_deflation":
                w = w - (B @ B.T) @ w
            elif self.source_deflation == "gram-schmidt":
                w = gram_schmidt(w, B)
            else:
                pass

            # Normalize
            w = w / np.linalg.norm(w)

            # Convergence criterion
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

        return w, k

    def mimimize_covisi(self, w, X, cov, fsamp):
        """
        Iterativly update a motor unit filter given a set of motor neuron
        spike times as long as the coefficient of variance of the interspike
        intervall decreases.

        Args:
            w (np.ndarray): Initial weight vector
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            cov (float): Coefficient of variance of the initial source
            fsamp (float): Sampling rate in Hz

        Returns:
            w (np.ndarray): Optimized weight vector
            spikes (np.ndarray): Sample indices of motor neuron discharges
            cov (float): Coefficient of variance of the optimized source

        """

        cov_last = cov + 1

        while cov < cov_last:
            source = w.T @ X
            spikes, _ = est_spike_times(source, fsamp)
            cov_last = cov
            isi = np.diff(spikes / fsamp)
            cov = np.std(isi) / np.mean(isi)
            w = np.mean(X[:, spikes], axis=1)
            w = w / np.linalg.norm(w)

        return w, spikes, cov

    def _write_pipeline_sidecar(self):
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass

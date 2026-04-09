import numpy as np
from numba import njit
from typing import List, Literal, Optional
from .core import bandpass_signals, notch_signals, extension, whitening, est_spike_times, remove_duplicates, remove_bad_sources, gram_schmidt, peel_off

class CBSS:
    """
    Class implementing convolutive blind source separation (CBSS) to identify the
    spiking activity of motor neurons using the fastICA algorithm.

    Parameters
    ----------
        random_seed (int): 
            Seed of the random number generator.
        ext_fact (int): 
            Extension factor
        whitening_method ("ZCA", "PCA" or "Cholesky"): 
            Method used for whitening
        whitening_regularization ("auto", float or None): 
            Method used to regularize small eigenvalues. If "auto" the mean of 
            the second half of the eigenvalues is used.
        cluster_method ("kmeans"): 
            Method used to seperate motor unit spikes and background spikes 
            (currently only "kmeans" is implemented).    
        ica_n_iter (int): 
            Number of fastICA runs, i.e., maximum number of extracted sources.
        opt_initalization ("random" or "activity_idx"): 
            Initalization method of the fastICA fixed-point algorithm. 
            Either drawn from a Gaussian distribution ("random") or using the 
            time instances with maximum column norms ("activity_idx").
        opt_function_exp (float): 
            Exponent a of the loss function g(x)=x * (x^2 + epsilon)^((a-1)/2) 
            representing a smooth approximation of g(x) = sign(x) * abs(x)**a. 
        opt_max_iter (int): 
            Maximum number of iterations for the fastICA fixed-point algorithm.  
        opt_tol (float): 
            Convergence criterion for the fixed-point algorithm. Stops if 
            the dot product between the current and previous unmixing weights 
            minus 1 is less than the tolerance value.
        source_deflation ("gram-schmidt", "projection_deflation" or None): 
            Method used to avoid repeaded convergence to the same source in 
            the fixed point algorithm.  
        refinement_loop (bool): 
            If True, the unmixing weights w are updated through self-supervised 
            learning. The updated unmixing weights are the mean of the 
            whitened signal at the detected spikes.
        refinement_loss ("cov_isi" or "sil"):
            Metric used as optimization loss in the refinement loop. Can be either
            minimizing the coefficient of variation (Cov) of the interspike 
            interalls ("cov_isi") or maximizing the silhouette score ("sil"). 
        refinement_max_iter (int):
            Maximum number of iterations of the refinement loop
        refinement_min_spikes (int): 
            Only enter the refinement loop if the number of detected spikes is 
            above the given threshold.        
        peel_off (bool): 
            If True, the contribution of identified sources is subtraced 
            from the whitened signal. 
        peel_off_window (float):
            Duration of the window used to peel off contributions from 
            detected sources (in seconds).    
        sil_th (float): 
            Classify sources into good (score above sil_th) or bad based 
            on a pseudo silhouette score.
        cov_th (float): 
            Classify sources into good (scores below cov_th) or bad based 
            on the coefficient of variation of the interspike intevalls.    
        verbose (bool): 
            If True, print progress.                

    Example
    -------
    Init CBSS class using the default parameters and run decomposition.
    >>> model = CBSS() 
    >>> sources, spikes, sil, mu_filters = model.decompose(sig=emg_data, fsamp=2048)


    """

    def __init__(
            self, 
            random_seed: int = 1909,
            ext_fact: int = 12,
            whitening_method: Literal["ZCA", "PCA", "Cholesky"] = "ZCA",
            whitening_reg: str | float | None = "auto",
            ica_n_iter: int = 100,
            opt_initalization: Literal["random", "activity_idx"] = "random",
            opt_function_exp: float = 3,
            opt_max_iter: int = 100,
            opt_tol: float = 1e-4,
            source_deflation: Literal["gram-schmidt", "projection_deflation", None] = "gram-schmidt",
            cluster_method: Literal["kmeans"] = "kmeans",
            peel_off: bool = True,
            peel_off_window: float = 0.025,
            sil_th: float = 0.9,
            cov_th: float = 0.35,
            refinement_loop: bool = True,
            refinement_loss: Literal["cov_isi", "sil"] = "cov_isi", 
            refinement_max_iter: int = 100,
            refinement_min_spikes: int = 10,
            unmixing_format: Literal["white", "ext"] = "white",
            match_th: float = 0.3,
            match_max_shift: float = 0.1,
            match_tol: float = 0.001,
            verbose: bool = True,
            config: dict | None = None, 
        ):

        # Default parameters
        self.bandpass = [20, 500]
        self.bandpass_order = 2
        self.notch_frequency = [50]
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1

        self.ext_fact = ext_fact
        self.whitening_method = whitening_method
        self.whitening_reg = whitening_reg
        self.ica_n_iter = ica_n_iter
        self.opt_initalization = opt_initalization
        self.opt_function_exp = opt_function_exp
        self.opt_max_iter = opt_max_iter
        self.opt_tol  = opt_tol
        self.source_deflation = source_deflation
        self.peel_off = peel_off
        self.peel_off_window = peel_off_window
        self.cluster_method = cluster_method
        self.random_seed = random_seed
        self.refinement_loop = refinement_loop
        self.refinement_min_spikes = refinement_min_spikes
        self.refinement_loss = refinement_loss
        self.refinement_max_iter = refinement_max_iter
        self.sil_th = sil_th
        self.cov_th = cov_th

        self.min_num_spikes = 10
        self.match_th = match_th
        self.match_max_shift = match_max_shift
        self.match_tol = match_tol
        self.verbose = verbose

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        valid_keys = self.__dict__.keys()

        # Assign all parameters as attributes
        for key, value in config_dict.items():
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

    def decompose(self, 
                  sig: np.ndarray, # (n_channels x n_samples)
                  fsamp: float
    ):
        """
        Run CBSS decomposition

        Args:
            sig (np.ndarray): Input (EMG) signal (n_channels x n_samples)
            fsamp (float): Sampling frequency in Hz

        Returns:
            sources (np.ndarray): Estimated spike responses (n_mu x n_samples)
            spikes (dict): Sample indices of motor neuron discharges
            sil (np.ndarray): Pseudo-silhouette scores of the estimated sources
            mu_filters (np.ndarray): Optimized motor unit filters
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
                freqs=self.notch_frequency,
                dfreq=self.notch_width,
                order=self.notch_order,
                #n_harmonics=self.notch_n_harmonics,
            )

        # Extend signals and subtract the mean and cut the edges
        print("Extension:")
        ext_sig = extension(sig, self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

        # Remove the edges from the exteneded signal
        ext_sig[:, : self.ext_fact * 2] = 0
        ext_sig[:, -self.ext_fact * 2 :] = 0
        print("  - Finished")

        # Whiten the extended signals
        print("Whitening:")
        white_sig, Z = whitening(
            Y=ext_sig, 
            method=self.whitening_method
        )
        print("  - Finished")

        # Initalize the output variables
        sources = np.zeros((self.ica_n_iter, sig.shape[1]))
        spikes = {i: [] for i in range(self.ica_n_iter)}
        sil = np.zeros(self.ica_n_iter)
        cov_isi = np.zeros(self.ica_n_iter)
        unmixing_weights = np.zeros((white_sig.shape[0], self.ica_n_iter))

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
            w, k1 = self._fixed_point_alg(w, white_sig, unmixing_weights)            

            # Predict source and estimate the source quality
            sources[i, :] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(
                sources[i, :], fsamp, cluster=self.cluster_method
            )
            cov_isi[i] = self._calc_cov_isi(spikes[i], fsamp)
            print(f"Iteration {i}:")
            print(f"  - Number of fixed point iterations: {k1}")
            print(f"  - SIL: {sil[i]}")
            print(f"  - cov_isi: {cov_isi[i]}")

            # Self supervised refinement loop
            if len(spikes[i]) > self.refinement_min_spikes and self.refinement_loop:
                w, k2 = self._self_supervised_refinement(
                    w, white_sig, sil[i], cov_isi[i], fsamp
                )
                sources[i, :] = w.T @ white_sig
                spikes[i], sil[i] = est_spike_times(
                    sources[i, :], fsamp, cluster=self.cluster_method
                )
                cov_isi[i] = self._calc_cov_isi(spikes[i], fsamp)
                print(f"    - Number of refinement iterations: {k2}")
                print(f"    - SIL: {sil[i]}")
                print(f"    - cov_isi: {cov_isi[i]}")

            # Save the optimized unmixing weights
            unmixing_weights[:, i] = w

            # Peel-off the detected source
            if self.peel_off and sil[i] > self.sil_th and cov_isi[i] < self.cov_th:
                white_sig, _, _ = peel_off(
                    white_sig, spikes[i], win=self.peel_off_window, fsamp=fsamp
                )
                print("    Peel off: True")

        # Remove duplicates
        sources, spikes, sil, unmixing_weights = remove_duplicates(
            sources,
            spikes,
            sil,
            unmixing_weights,
            fsamp,
            max_shift=self.match_max_shift,
            tol=self.match_tol,
            threshold=self.match_th,
        )

        # Remove bad sources
        sources, spikes, sil, unmixing_weights = remove_bad_sources(
            sources,
            spikes,
            sil,
            unmixing_weights,
            threshold=self.sil_th,
            min_num_spikes=self.min_num_spikes,
        )

        return sources, spikes, sil, unmixing_weights

    def _fixed_point_alg(self, w, X, B, epsilon=1e-3):
        """
        Fixed-point optimization to maximize sparseness of a source signal.
        The optimization function is G(x)= x * (x^2+epsilon)^{(a-1)/2} that represents 
        a smooth approximation of G(x) = sign(x) * abs(x)^a.
        For a = 3 this is equvivalent to maximizing skewness.

        Args:
            w (np.ndarray): Initial weight vector (n_channels,)
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            B (np.ndarray): Current separation matrix (n_components x n_channels)

        Returns:
            w (np.ndarray): Optimized weight vector
            k (int): Number of iterations taken
        """

        delta = np.ones(self.opt_max_iter)
        k = 0

        while delta[k] > self.opt_tol and k < self.opt_max_iter - 1:
            w_last = w.copy()

            wTX = w.T @ X  # shape: (n_samples,)
            # First derivative G'(x)
            g = (
                (epsilon + wTX**2) ** ((self.opt_function_exp - 3) / 2) 
                * (self.opt_function_exp * wTX**2 + epsilon)
            )
            # Second derivative G''(x)
            gp = (
                (self.opt_function_exp - 1)
                * wTX
                * (epsilon + wTX**2) ** ((self.opt_function_exp - 5) / 2)
                * (self.opt_function_exp * wTX**2 + 3 * epsilon)
            )
            A = np.mean(gp)
            w = np.mean(X * g, axis=1) - A * w  # shape: (n_channels,)

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

    def _self_supervised_refinement(self, w, X, sil, cov_isi, fsamp):
        """
        Iterativly update a motor unit filter given a set of motor neuron
        spike times as long as the coefficient of variance of the interspike
        intervall decreases.

        Args
        ----
            w (np.ndarray): Initial weight vector
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            cov (float): Coefficient of variance of the initial source
            fsamp (float): Sampling rate in Hz

        Returns
        -------
            w (np.ndarray): Optimized weight vector

        """

        # Init the optimization
        score = self._get_refinement_loss(sil, cov_isi)
        score_last = score + 1
        k = 0

        while score < score_last and k < self.refinement_max_iter:
            source = w.T @ X
            spikes, sil = est_spike_times(source, fsamp)
            score_last = score
            cov_isi = self._calc_cov_isi(spikes, fsamp)
            score = self._get_refinement_loss(sil, cov_isi)
            w = np.mean(X[:, spikes], axis=1)
            w = w / np.linalg.norm(w)
            k += 1

        return w, k
    
    def _calc_cov_isi(self, spikes, fsamp):
        """
        Helper function to calculate the coefficent of varation
        of the interspike intervalls
        
        """

        if len(spikes) > 2:
            isi = np.diff(spikes / fsamp)
            cov_isi = np.std(isi) / np.mean(isi)
        else:
            cov_isi = np.inf

        return cov_isi
    
    def _get_refinement_loss(self, sil, cov_isi):
        """
        Helper function to compute the loss in the refinement loop
        
        """

        if self.refinement_loss == "cov_isi":
            score = cov_isi
        else:
            score = 1 - sil

        return score


    def _write_pipeline_sidecar(self):
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass

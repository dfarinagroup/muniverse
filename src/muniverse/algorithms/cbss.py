import numpy as np
from typing import List, Literal, Optional
from .core import (
    extension, 
    whitening, 
    est_spike_times, 
    gram_schmidt, 
    peel_off,
    spike_dict_to_long_df
)

class _BaseCBSS():
    """
    
    Base class for CBSS-based motor unit idendification

    Properties
    ----------

        ext_fact : int , default 12
            Extension factor

        whitening_method : {"ZCA", "PCA", "Cholesky"}, default "ZCA" 
            Method used for whitening

        whitening_regularization : {"auto", float, None}, default "auto" 
            Adds a small value to the eigenvalues for regularization. 
            If "auto", the mean of the second half of the eigenvalues is used.

        whitening_backend : {"ed", "svd"}, default "ed" 
            Method used to calculate eigenvalues and eigenvectors. Can be
            either based on singular value decomposition ("svd") or an
            eigendecomposition ("ed"). Only needed if method is "ZCA" or "PCA".    

        spike_detection_exp : float , default 2
            Exponent of asymetric power law applied to the extracted sources
            before spike detection

        spike_detection_min_delay : float , default 0.01
            Minimum distance between two detected spikes in seconds  

        verbose : bool , default True
            Verbose mode     
    
    """

    def __init__(
            self,
            ext_fact: int = 12,
            whitening_method: Literal["ZCA", "PCA", "Cholesky"] = "ZCA",
            whitening_backend: Literal["ed, svd"] = "ed",
            whitening_reg: str | float | None = "auto",
            spike_detection_exp: float = 2,
            spike_detection_min_delay: float = 0.01,
            verbose: bool = False
    ):
        # Default parameters
        self.ext_fact = ext_fact
        self.whitening_method = whitening_method
        self.whitening_backend = whitening_backend
        self.whitening_reg = whitening_reg
        self.spike_detection_exp = spike_detection_exp
        self.spike_detection_min_delay = spike_detection_min_delay
        self.verbose = verbose

        self._params = set(self.__dict__.keys()) - {"_params"}

        self._attributes = set([
            "unmixing_weights_", "whiten_", "unwhiten_"
        ])

    def set_parameter(self, **kwargs):
        """
        Update parameters given an arbitary list of key value pairs

        Args
        ----
            **kwargs
                Parsed parameteters
        
        """
        for key, value in kwargs.items():
            if key in self._params:
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")        


    def _extension(self, sig: np.ndarray):
        """Extend the data, subtract the mean and cut the edges"""

        if self.verbose:
            print("Step: Extension")
            print(f"  - Extension factor: {self.ext_fact}")

        ext_sig = extension(sig, self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)
        ext_sig[:, : self.ext_fact * 2] = 0
        ext_sig[:, -self.ext_fact * 2 :] = 0

        return ext_sig

    def _whitening(self, data: np.ndarray, return_data=True):
        """Whiten the data matrix"""

        if self.verbose:
            print("Step: Whitening")
            print(f"  - Method: {self.whitening_method}")        

        white_sig, self.whiten_, self.unwhiten_ = whitening(
            Y = data, 
            method = self.whitening_method,
            backend = self.whitening_backend, 
            regularization = self.whitening_reg 
        )

        if return_data:
            return white_sig
    
    def _calc_cov_isi(self, spikes, fsamp):
        """ Get the coefficent of varation of the interspike intervalls """

        if len(spikes) > 2:
            isi = np.diff(spikes / fsamp)
            cov_isi = np.std(isi) / np.mean(isi)
        else:
            cov_isi = np.inf

        return cov_isi
    
    def predict(
            self, 
            sig: np.ndarray, # (n_channels x n_samples)
            fsamp: float
    ):
        """
        Predict motor unit spike trains given multi-channel 
        EMG data using your learned unmixing weights.

        Args
        ----
            sig : np.ndarray 
                Input (EMG) signal (n_channels x n_samples)

            fsamp : float 
                Sampling frequency in Hz  

        Returns
        -------
            spikes : pd.DataFrame 
                Spike table (columns: onset, duration, sample, unit_id, description)

            sources : np.ndarray 
                Estimated sources / ica components (n_components x n_samples)

            scores : dict of np.ndarray 
                Source trustworthiness scores ("sil" and "cov_isi")
        
        """

        if not hasattr(self, "unmixing_weights_"):
            raise ValueError(
                "No unmixing weights are defined."
                "Make sure to fit the model before calling this function."
            )

        # Extend signals and subtract the mean and cut the edges
        ext_sig = self._extension(sig)

        # Whiten data
        white_sig = self.whiten_ @ ext_sig

        # Apply unmixing weidths
        sources = self.unmixing_weights_.T @ white_sig

        # Init spikes and scores
        spikes = {i: [] for i in range(sources.shape[0])}
        scores = {
            "sil": np.zeros(sources.shape[0]),
            "cov_isi": np.zeros(sources.shape[0]),
        }

        # Get spikes and scores
        for i in range(sources.shape[0]):
            spikes[i], scores["sil"][i] = est_spike_times(
                source = sources[i, :], 
                fsamp = fsamp, 
                a = self.spike_detection_exp,
                min_delay = self.spike_detection_min_delay
            )
            scores["cov_isi"][i] = self._calc_cov_isi(spikes[i], fsamp)

        # Convert dict of spikes to long-formated spike table 
        spikes = spike_dict_to_long_df(spikes)

        return spikes, sources, scores 
    
    def save_model(self):
        """"
        Save the model parameters and learned attributes
        for downstream usage
        
        """

        params = {}
        attributes = {}
        for key, value in self.__dict__.items():
            if key in self._params:
                params[key] = value
            elif key in self._attributes:
                attributes[key] = value 
        
        return {
            "parameters": params,
            "attributes": attributes
        }
    
    def load_model(self, model):
        """"
        Load a model including both parameters and learned 
        attributes for downstream usage

        Args
        ----
            model : dict
                Dictonary containing all parameters and learned 
                attributes of the model 
        
        """

        for key, value in model["parameters"].items():
            if key in self._params:
                setattr(self, key, value)
        for key, value in model["attributes"].items():        
            if key in self._attributes:
                setattr(self, key, value)

class FastIcaCBSS(_BaseCBSS):
    """
    Class implementing convolutive blind source separation (CBSS) 
    to identify the spiking activity of motor neurons given
    multi-channel EMG data. The algorithm performs the following steps:

    - Transform the convolutive mixture into an instantaneous mixture 
    by adding R delayed copies of the input signal
    - Apply a whitening transformation to the extended signals to 
    obtain data with unit variance
    - Sequentially optimize the unmixing weights using the fastICA 
    algorithm
        - Run a fixed-point algorithm maximizing non-gaussianity of
        the sources
        - Extract the motor unit spikes using peak-detection and 
        kmeans++ based spike clustering
        - Refine the learned weights through self-supervised 
        learning (optional)
        - Peel-off the contribution of detected sources (optional)

    Properties
    ----------

        random_seed : int , default 1909 
            Seed of the random number generator.

        ext_fact : int , default 12
            Number of delayed copies in the extended signal

        whitening_method : {"ZCA", "PCA", "Cholesky"}, default "ZCA" 
            Method used for whitening

        whitening_backend : {"ed", "svd"}, default "ed" 
            Method used to calculate eigenvalues and eigenvectors. Can be
            either based on singular value decomposition ("svd") or an
            eigendecomposition ("ed"). Only needed if method is "ZCA" or "PCA".    

        whitening_regularization : {"auto", float, None}, default "auto" 
            Adds a small value to the eigenvalues for regularization. 
            If "auto", the mean of the second half of the eigenvalues is used.

        spike_detection_exp : float , default 2
            Exponent of asymetric power law applied to the extracted sources
            before spike detection
            
        spike_detection_min_delay : float , default 0.01
            Minimum distance between two detected spikes in seconds   

        ica_iterations : int , default 100
            Number of fastICA runs, i.e., maximum number of extracted sources.

        ica_initalization : {"random", "activity_idx"}, default "random" 
            Initalization method of the fastICA fixed-point algorithm. 
            Either drawn from a Gaussian distribution ("random") or using the 
            time instances with maximum column norms ("activity_idx").

        ica_opt_fun_exp : float , default 3
            Exponent a of the loss function g(x)=x * (x^2 + epsilon)^((a-1)/2) 
            representing a smooth approximation of g(x) = sign(x) * abs(x)**a. 

        ica_opt_fun_eps : float , default 1e-3
            Epsilon used in the loss function.    

        ica_max_iter : int , default 100
            Maximum number of iterations for the fastICA fixed-point algorithm.  

        ica_tol : float , default 5e-4 
            Convergence criterion for the fastICA fixed-point algorithm. Stops if 
            the dot product between the current and previous unmixing weights 
            minus 1 is less than the tolerance value.

        ica_orthogonalization : {"gram-schmidt", "projection_deflation",  None}, default "gram-schmidt"
            Method used to avoid repeaded convergence to the same source in 
            the fastICA fixed point algorithm.  

        refinement_loop : bool , default True
            If True, the unmixing weights w are updated through self-supervised 
            learning. The updated unmixing weights are the mean of the 
            whitened signal at the detected spikes.

        refinement_loss : {"cov_isi", "sil"}, default "cov_isi"
            Metric used as optimization loss in the refinement loop. Can be either
            minimizing the coefficient of variation (Cov) of the interspike 
            interalls ("cov_isi") or maximizing the silhouette score ("sil"). 

        refinement_max_iter : int , default 100
            Maximum number of iterations of the refinement loop

        refinement_min_spikes : int , default 10
            Only enter the refinement loop if the number of detected spikes is 
            above the given threshold.        

        peel_off : bool , default True
            If True, the contribution of identified sources is subtraced 
            from the whitened signal. 

        peel_off_window : float , default 0.025
            Duration of the window used to peel off contributions from 
            detected sources (in seconds).    

        peel_of_sil_th : float , default 0.9
            Only apply peel off if the source score is above the given
            threshold value

        peel_of_cov_th : float , default 0.35
            Only apply peel off if the source score is below the given
            threshold value

        verbose : bool , default True
            Verbose mode

    Attributes
    ----------

        unmixing_weights_ : np.ndarray (n_features, n_components)
            The learned unmixing weights

        whiten_ : np.ndarray (n_features, n_features)
            Whitening matrix   

        unwhiten_ : np.ndarray (n_features, n_features)
            Inverse of the whitening matrix    

        n_fixed_point_iter_ : np.ndarray of int
            Number of iterations in the fixed point algorithm  

        fixed_point_deltas_ : np.ndarray
            Cosine distance between two fixed point updates

        n_refinement_iter_ : np.ndarray of int
            Number of iterations in the fixed point algorithm  

        refinement_scores_ : np.ndarray
            Score at each refinement iteration

        peel_off_ : np.ndarray of bool
            Whether a source was peeled off or not

    References
    ----------
    .. [1] Negro et et al., "Multi-channel intramuscular and surface
           EMG decomposition by convolutive blind source separation",
           Journal of Neural Engineering, 2016 
    .. [2] Mamidanna et et al., "MUniverse: A Simulation and Benchmarking 
           Suite for Motor Unit Decomposition", The Thirty-ninth Annual 
           Conference on Neural Information Processing Systems 
           Datasets and Benchmarks Track, 2025                    


    Example
    -------

    Init CBSS class using the default parameters and run decomposition.
    >>> model = CBSS() 
    >>> spikes, sources, scores = model.fit_predict(sig=emg_data, fsamp=2048)


    """

    def __init__(
            self, 
            random_seed: int = 1909,
            ext_fact: int = 12,
            whitening_method: Literal["ZCA", "PCA", "Cholesky"] = "ZCA",
            whitening_backend: Literal["ed", "svd"] = "ed",
            whitening_reg: str | float | None = "auto",
            spike_detection_exp: float = 2,
            spike_detection_min_delay: float = 0.01,
            #spike_cluster_method: Literal["kmeans"] = "kmeans",
            ica_iterations: int = 100,
            ica_initalization: Literal["random", "activity_idx"] = "random",
            ica_opt_fun_exp: float = 3,
            ica_opt_fun_eps: float = 1e-3,
            ica_max_iter: int = 100,
            ica_tol: float = 5e-4,
            ica_orthogonalization: Literal["gram-schmidt", "projection_deflation", None] = "gram-schmidt",
            refinement_loop: bool = True,
            refinement_loss: Literal["cov_isi", "sil"] = "cov_isi", 
            refinement_max_iter: int = 100,
            refinement_min_spikes: int = 10,
            peel_off: bool = True,
            peel_off_window: float = 0.025,
            peel_off_sil_th: float = 0.9,
            peel_off_cov_th: float = 0.35,
            verbose: bool = True,
            config: dict | None = None, 
        ):

        super().__init__(
            ext_fact = ext_fact,
            whitening_method = whitening_method,
            whitening_backend = whitening_backend, 
            whitening_reg = whitening_reg,
            spike_detection_exp = spike_detection_exp,
            spike_detection_min_delay = spike_detection_min_delay,
            verbose = verbose
        )

        self.random_seed = random_seed
        self.ica_iterations = ica_iterations
        self.ica_initalization = ica_initalization
        self.ica_opt_fun_exp = ica_opt_fun_exp
        self.ica_opt_fun_eps = ica_opt_fun_eps
        self.ica_max_iter = ica_max_iter
        self.ica_tol  = ica_tol
        self.ica_orthogonalization = ica_orthogonalization
        self.refinement_loop = refinement_loop
        self.refinement_min_spikes = refinement_min_spikes
        self.refinement_loss = refinement_loss
        self.refinement_max_iter = refinement_max_iter
        self.peel_off = peel_off
        self.peel_off_window = peel_off_window
        self.peel_off_sil_th = peel_off_sil_th
        self.peel_off_cov_th = peel_off_cov_th

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        self._params = set(self.__dict__.keys()) - {"_params"}

        # Set all parameters from the config dict
        for key, value in config_dict.items():
            if key in self._params:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

        self._attributes = set([
            "unmixing_weights_", "whiten_", "unwhiten_",
            "n_fixed_point_iter_", "fixed_point_deltas_",
            "n_refinement_iter_", "refinement_scores_", "peel_off_"
        ])      

    def fit_predict(self, 
                  sig: np.ndarray, # (n_channels x n_samples)
                  fsamp: float
    ):
        """
        Run CBSS decomposition

        Args
        ----
            sig : np.ndarray 
                Input (EMG) signal (n_channels x n_samples)

            fsamp : float 
                Sampling frequency in Hz

        Returns
        -------
            spikes : dict 
                Sample indices of motor neuron discharges

            sources : np.ndarray 
                Estimated sources / ica components (n_components x n_samples)
            
            scores : dict
                Dictonary of source quality scores ("sil" and "cov_isi")

        """

        # Initalize random number generator
        rng = np.random.seed(self.random_seed)

        # Extend signals and subtract the mean and cut the edges
        ext_sig = self._extension(sig)

        # Whiten the extended signals
        white_sig = self._whitening(ext_sig)

        # Initalize the output variables
        sources = np.zeros((self.ica_iterations, sig.shape[1]))
        spikes = {i: [] for i in range(self.ica_iterations)}
        scores = {
            "sil": np.zeros(self.ica_iterations),
            "cov_isi": np.zeros(self.ica_iterations),
            "ica_target": np.zeros(self.ica_iterations)
        }
        self.n_fixed_point_iter_ = np.zeros(self.ica_iterations, dtype=int)
        self.fixed_point_deltas_ = {i: [] for i in range(self.ica_iterations)}
        self.n_refinement_iter_ = np.zeros(self.ica_iterations, dtype=int)
        self.refinement_scores_ = {i: [] for i in range(self.ica_iterations)}
        self.peel_off_ = np.zeros(self.ica_iterations, dtype=bool)
        self.unmixing_weights_ = np.zeros((white_sig.shape[0], self.ica_iterations))

        if self.ica_initalization == "activity_idx":
            act_idx_histoty = np.array([])

        # Loop over each MU
        for i in range(self.ica_iterations):
            if self.verbose:
                print(f'Step: FastICA iteration {i}:')
            # Initalize
            if self.ica_initalization == "random":
                w = np.random.randn(white_sig.shape[0])
            elif self.ica_initalization == "activity_idx":
                col_norms = np.linalg.norm(white_sig, axis=0)
                col_norms[act_idx_histoty.astype(int)] = 0
                best_idx = np.argmax(col_norms)
                w = white_sig[:, best_idx]
                act_idx_histoty = np.append(act_idx_histoty, best_idx)
            else:
                ValueError("The specified initalization method is not implemented")

            # fastICA fixedpoint optimization
            w = self._fixed_point_alg(w, white_sig, i)         

            # Predict source and estimate the source quality
            sources[i, :] = w.T @ white_sig
            spikes[i], scores["sil"][i] = est_spike_times(
                source = sources[i, :], 
                fsamp = fsamp, 
                a = self.spike_detection_exp,
                min_delay = self.spike_detection_min_delay
            )
            scores["cov_isi"][i] = self._calc_cov_isi(spikes[i], fsamp)
            scores["ica_target"][i] = self._get_ica_loss(sources[i,:])

            # Self supervised refinement loop
            if len(spikes[i]) > self.refinement_min_spikes and self.refinement_loop:
                scores_i = {k: v[i] for k, v in scores.items()}
                w = self._self_supervised_refinement(
                    w, white_sig, scores_i, fsamp, i
                )
                sources[i, :] = w.T @ white_sig
                spikes[i], scores["sil"][i] = est_spike_times(
                    source = sources[i, :], 
                    fsamp = fsamp, 
                    a = self.spike_detection_exp,
                    min_delay = self.spike_detection_min_delay
                )
                scores["cov_isi"][i] = self._calc_cov_isi(spikes[i], fsamp)
                scores["ica_target"][i] = self._get_ica_loss(sources[i,:])

            # Save the optimized unmixing weights
            self.unmixing_weights_[:, i] = w

            if self.verbose:
                print(f'  - SIL: {scores["sil"][i]}')
                print(f'  - cov_isi: {scores["cov_isi"][i]}')

            # Peel-off the detected source
            if (
                self.peel_off 
                and scores["sil"][i] > self.peel_off_sil_th 
                and scores["cov_isi"][i] < self.peel_off_cov_th
                ):
                white_sig, _, _ = peel_off(
                    white_sig, spikes[i], win=self.peel_off_window, fsamp=fsamp
                )
                self.peel_off_[i] = True
                if self.verbose:
                    print('  - Peel off: True')
            else:
                if self.verbose:
                    print('  - Peel off: False')

        # Convert dict of spikes to long-formated spike table 
        spikes = spike_dict_to_long_df(spikes)

        return spikes, sources, scores
            
    def _fixed_point_alg(self, w, X, i):
        """
        Fixed-point algorithm to maximize sparseness of a source signal.
        The optimization function is:: 
            G(x)= x * (x^2+epsilon)^{(a-1)/2} ,
        that represents a smooth approximation of:: 
            G(x) = sign(x) * abs(x)^a .
        For a = 3 this is equvivalent to maximizing skewness.

        Args
        ----
            w : np.ndarray
                Initial unmixing weight vector (n_channels,)
            X : np.ndarray 
                Whitened signal (n_channels x n_samples)
            i : int
                The current iteration     

        Returns
        -------
            w : np.ndarray 
                Optimized unmixing weight vector
        """

        if self.ica_orthogonalization == "projection_deflation":
            P = self.unmixing_weights_ @ self.unmixing_weights_.T

        delta = np.ones(self.ica_max_iter)
        k = 0

        while delta[k] > self.ica_tol and k < self.ica_max_iter - 1:
            w_last = w.copy()

            wTX = w.T @ X  # shape: (n_samples,)
            # First derivative G'(x)
            g = (
                (self.ica_opt_fun_eps + wTX**2) ** ((self.ica_opt_fun_exp - 3) / 2) 
                * (self.ica_opt_fun_exp * wTX**2 + self.ica_opt_fun_eps)
            )
            # Second derivative G''(x)
            gp = (
                (self.ica_opt_fun_exp - 1)
                * wTX
                * (self.ica_opt_fun_eps + wTX**2) ** ((self.ica_opt_fun_exp - 5) / 2)
                * (self.ica_opt_fun_exp * wTX**2 + 3 * self.ica_opt_fun_eps)
            )
            A = np.mean(gp)
            w = np.mean(X * g, axis=1) - A * w  # shape: (n_channels,)

            # Orthogonalization step
            if self.ica_orthogonalization == "projection_deflation":
                w = w - P @ w
            elif self.ica_orthogonalization == "gram-schmidt":
                w = gram_schmidt(w, self.unmixing_weights_)
            else:
                pass

            # Normalize
            w = w / np.linalg.norm(w)

            # Convergence criterion
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

        self.n_fixed_point_iter_[i] = k 
        self.fixed_point_deltas_[i] = delta[:k]

        if self.verbose:
            print(f'  - Number of fixed point iterations: {k}')

        return w

    def _self_supervised_refinement(self, w, X, scores_i, fsamp, i):
        """
        Iterativly update a motor unit filter given a set of motor neuron
        spike times as long as the coefficient of variance of the interspike
        intervall decreases.

        Args
        ----
            w : np.ndarray 
                Initial unmixing weight vector
            X : np.ndarray 
                Whitened signal matrix (n_channels x n_samples)
            scores_i : dict 
                Dictonary of quality scores at iteration i {"name": value}
            fsamp : float 
                Sampling rate in Hz

        Returns
        -------
            w : np.ndarray 
                Optimized weight vector

        """

        # Init the optimization
        scores = np.zeros(self.refinement_max_iter)
        score_last = np.inf
        k = 0

        while k < self.refinement_max_iter:
            source = w.T @ X
            spikes, sil = est_spike_times(
                source = source, 
                fsamp = fsamp, 
                a = self.spike_detection_exp,
                min_delay = self.spike_detection_min_delay
            )

            cov_isi = self._calc_cov_isi(spikes, fsamp)
            scores_i = {
                "sil": sil, 
                "cov_isi": cov_isi
            }
            scores[k] = self._get_refinement_loss(scores_i)

            if score_last - scores[k] <= 1e-4:
                break

            score_last = scores[k]

            w = np.mean(X[:, spikes], axis=1)
            w = w / np.linalg.norm(w)
            k += 1

        self.n_refinement_iter_[i] = k 
        self.refinement_scores_[i] = scores[:k]

        if self.verbose:
            print(f'  - Number of refinement iterations: {k}')

        return w
        
    def _get_refinement_loss(self, scores_i):
        """ Compute the loss in the refinement loop """

        if self.refinement_loss == "cov_isi":
            score = scores_i["cov_isi"]
        else:
            score = 1 - scores_i["sil"]

        return score
    
    def _get_ica_loss(self, wTX):
        """Compute the score of the ICA optimization function"""
        G = np.mean(
                wTX * (
                wTX**2 + self.ica_opt_fun_eps
                )**((self.ica_opt_fun_exp - 1) / 2)
            )
        
        return G
    
        
       

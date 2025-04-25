import numpy as np
from decomposition_routines import *
import sys

class upper_bound:

    def __init__(self, config=None, **kwargs):
        self.ext_fact = 12
        self.whitening_method = 'ZCA'
        self.whitening_reg  = 'auto'
        self.cluster_method  = 'kmeans'

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

        Parameters:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            muaps (ndarray): MUAPs (mu_index x n_channels x duration)
            fsamp (float): Sampling rate of the data (unit: Hz)

        Returns:
            sources (np.ndarray): Estimated sources (n_mu x n_samples)
            spikes (dict): Spiking instances of the motor neurons
            sil (np.ndarray): Source quality metric
        """

        n_mu  = muaps.shape[0]
        sources  = np.zeros((n_mu, sig.shape[1]))
        spikes = {i: [] for i in range(n_mu)}
        sil = np.zeros(n_mu)

        # Extend signals and subtract the mean
        ext_sig = extension(sig, self.ext_fact)
        ext_mean = np.mean(ext_sig, axis=1, keepdims=True) 
        ext_sig -= ext_mean

        # Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method)

        # Loop over each MU
        for i in np.arange(n_mu):
            # Get the optimal MU filter
            w = self.muap_to_filter(muaps[i,:,:], ext_mean, Z)
            # Estimate source
            sources[i,:] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(sources[i,:], fsamp, cluster=self.cluster_method)

        return sources, spikes, sil


    def muap_to_filter(self, muap, ext_mean, Z):
        """
        Get the optimal motor unit filter from the ground truth MUAP.
        Therefore, the MUAP is extended and whitened. The optimal motor unit
        filter corresponds to the column of the extended and whitened MUAP
        that has the highest norm.

        Parameters:
            MUAP (ndarray): Multichannel MUAP (n_channels x duration)
            Z (ndarray): Whitening matrix

        Returns:
            w (ndarray): Normalized motor unit filter
        """

        # Extend the MUAP
        ext_muap = extension(muap,self.ext_fact) 
        ext_muap -= ext_mean

        # Whiten the MUAP
        white_muap = Z @ ext_muap

        # Find the column with the largest L2 norm and return it as MUAP filter
        col_norms = np.linalg.norm(white_muap, axis=0)
        w = white_muap[:, np.argmax(col_norms)]

        # Normalize w
        w = w/np.linalg.norm(w)

        return(w)
    

class basic_cBSS:

    def __init__(self, config=None, **kwargs):

        # Default parameters
        self.ext_fact = 12
        self.whitening_method = 'ZCA'
        self.whitening_reg  = 'auto'
        self.n_iter = 100
        self.opt_initalization = 'random'
        self.opt_function_exp = 3
        self.opt_max_iter = 100
        self.opt_tol = 1e-4
        self.source_deflation = 'gram-schmidt'
        self.peel_off = 'false'
        self.cluster_method  = 'kmeans'
        self.random_seed = 1909
        self.refinement_loop = False
        self.sil_th = 0.9
        self.cov_th = 0.35

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

        Parameters:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            fsamp (float): Sampling frequency in Hz

        Returns:
            sources (ndarray): Estimated spike responses (n_mu x n_samples)
            spikes (dict): Sample indices of motor neuron discharges 
            mu_filters (ndarray): Optimized motor unit filters
        """

        # Initalize random number generator
        rng = np.random.seed(self.random_seed)

        # Extend signals and subtract the mean and cut the edges
        ext_sig = extension(sig,self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

        # Remove the edges from the exteneded signal
        ext_sig[:,:self.ext_fact*2] = 0
        ext_sig[:,-self.ext_fact*2:] = 0


        # Step 1.3: Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method)

        sources  = np.zeros((self.n_iter,sig.shape[1]))
        spikes = {i: [] for i in range(self.n_iter)}
        sil = np.zeros(self.n_iter)
        B = np.zeros((white_sig.shape[0], self.n_iter))

        if self.opt_initalization == 'activity_idx':
            act_idx_histoty = np.zeros(self.n_iter).astype(bool)

        # Loop over each MU
        for i in np.arange(self.n_iter):
            # Initalize 
            if self.opt_initalization == 'random':
                w = np.random.randn(white_sig.shape[0])
            elif self.opt_initalization == 'activity_idx':
                col_norms = np.linalg.norm(white_sig, axis=0)
                col_norms[act_idx_histoty] = 0
                best_idx = np.argmax(col_norms)
                w = white_sig[:, best_idx]
                act_idx_histoty[i] = best_idx
            else:
                ValueError('The specified initalization method is not implemented')
            # fastICA fixedpoint optimization
            w, _ = self.my_fixed_point_alg(w, white_sig, B)
            # Predict source and estimate the source quality
            sources[i,:] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(sources[i,:], fsamp, cluster=self.cluster_method)
            isi      = np.diff(spikes[i]/fsamp)
            cov  = np.std(isi) / np.mean(isi)
            # Refinement loop
            if len(spikes[i]) > 10 and self.refinement_loop:
                w, _, cov = self.mimimize_covisi(w,white_sig, cov, fsamp)
                sources[i,:] = w.T @ white_sig
                spikes[i], sil[i] = est_spike_times(sources[i,:], fsamp, cluster=self.cluster_method)

            B[:,i] = w

            if self.peel_off and sil[i] > self.sil_th and cov < self.cov_th:
                white_sig, _ = peel_off(white_sig, spikes[i], win=0.025, fsamp=fsamp) 

        # Remove dublicates        
        sources, spikes, sil = remove_duplicates(sources, spikes, sil, fsamp)
       
        return sources, spikes, sil


    def my_fixed_point_alg(self, w, X, B):
        """
        Fixed-point optimization to maximize sparseness of a source signal.

        Parameters:
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
        g = lambda x: (epsilon+x**2)**(a-3/2) * (2*a*x**2 + epsilon)
        gp = lambda x: (2*a-1)*x * (epsilon+x**2)**(a-5/2) * (2*a*x**2 + 3*epsilon)

        TOL = self.opt_tol
        delta = np.ones(self.opt_max_iter)
        k = 0

        while delta[k] > TOL and k < self.opt_max_iter - 1:
            w_last = w.copy()

            wTX = w.T @ X  # shape: (n_samples,)
            A = np.mean(gp(wTX))
            w = np.mean(X * g(wTX), axis=1) - A * w  # shape: (n_channels,)

            # Orthogonalization step
            if self.source_deflation == 'projection_deflation':
                w = w - (B @ B.T) @ w
            elif self.source_deflation == 'gram-schmidt':
                w = gram_schmidt(w, B)
            else:
                pass            

            # Normalize
            w = w / np.linalg.norm(w)

            # Convergence criterion
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

        return w, k    
    
    def mimimize_covisi(self, w, X,  cov, fsamp):

        cov_last = cov + 1

        while cov < cov_last:
            source = w.T @ X
            spikes, _ = est_spike_times(source, fsamp)
            cov_last = cov
            isi      = np.diff(spikes/fsamp)
            cov  = np.std(isi) / np.mean(isi)
            w = np.mean(X[:,spikes],axis=1) 
            w = w / np.linalg.norm(w)
         
        return w, spikes, cov
    
    def _write_pipeline_sidecar(self):
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass
    


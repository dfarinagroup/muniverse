import numpy as np
from decomposition_routines import *

class upper_bound:

    def __init__(self):
        self.ext_fact = 12
        self.whitening_method = 'ZCA'
        self.whitening_reg  = 'auto'
        self.cluster_method  = 'kmeans'

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

    def __init__(self):
        self.ext_fact = 12
        self.whitening_method = 'ZCA'
        self.whitening_reg  = 'auto'
        self.n_iter = 50
        self.obj_function = 'skew'
        self.max_iter = 100
        self.deflation = 'deflation'
        self.peel_off = 'false'
        self.cluster_method  = 'kmeans'

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

        # Loop over each MU
        for i in np.arange(self.n_iter):
            # Get the optimal MU filter
            w = np.random.randn(white_sig.shape[0])
            w, k = self.my_fixed_point_alg(w, white_sig, B)
            # Estimate source
            sources[i,:] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(ipts[i,:], fsamp, cluster=self.cluster_method)
            B[:,i] = w

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

    def gram_schmidt(v, B):
        """Project v orthogonally to all vectors in B."""
        for b in B:
            v = v - np.dot(v, b) * b
        return v

    # Define contrast function and its derivative
    if self.obj_function == 'skew':
        g = lambda x: x**2
        gp = lambda x: 2 * x
    elif self.obj_function == 'kurtosis':
        g = lambda x: x**3
        gp = lambda x: 3 * x**2
    elif self.obj_function == 'logcosh':
        g = lambda x: np.log(np.cosh(x))
        gp = lambda x: np.tanh(x)
    else:
        raise ValueError(f"Unknown contrast function: {self.obj_function}")

    TOL = 1e-4
    delta = np.ones(self.maxiter)
    k = 0

    while delta[k] > TOL and k < self.maxiter - 1:
        w_last = w.copy()

        wTX = w.T @ X  # shape: (n_samples,)
        A = np.mean(gp(wTX))
        w = np.mean(X * g(wTX), axis=1) - A * w  # shape: (n_channels,)

        # Orthogonalization step
        if self.deflation == 'deflation':
            w = w - (B @ B.T) @ w
        elif self.deflation == 'gram-schmid':
            w = gram_schmidt(w, B)
        elif self.deflation == 'none':
            pass
        else:
            raise ValueError(f"Unknown orthogonalization: {self.deflation}")

        # Normalize
        w = w / np.linalg.norm(w)

        # Convergence criterion
        delta[k + 1] = abs(np.dot(w, w_last) - 1)
        k += 1

    return w, k    
    


import numpy as np
from decomposition_routines import *
from source_evaluation import est_spike_times

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
    


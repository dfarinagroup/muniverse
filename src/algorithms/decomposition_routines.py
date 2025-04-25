import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import find_peaks, convolve
from sklearn.cluster import KMeans
import sys
sys.path.append('../evaluation/')
from evaluate import *

def extension(Y, R):
    """
    Extend a multi-channel signal Y by an extension factor R
    using Toeplitz matrices.

    Parameters:
        Y (ndarray): Original signal (n_channels x n_samples)
        R (int): Extension factor (number of lags)

    Returns:
        eY (ndarray): Extended signal (n_channels * R x n_samples)
    """
    n_channels, n_samples = Y.shape
    eY = np.zeros((n_channels * R, n_samples))

    for i in range(n_channels):
        col = np.concatenate(([Y[i, 0]], np.zeros(R - 1)))
        row = Y[i, :]
        T = toeplitz(col, row)
        eY[i * R:(i + 1) * R, :] = T

    return eY

def whitening(Y, method='ZCA', backend='eig', regularization='auto', eps=1e-10):
    """
    Adaptive whitening function using ZCA, PCA, or Cholesky.

    Parameters:
        Y (ndarray): Input signal (n_channels x n_samples)
        method (str): Whitening method: 'ZCA', 'PCA', 'Cholesky'
        backend (str): 'eig', or 'svd'
        regularization (str or float): 'auto', float value, or None
        eps (float): Small epsilon for numerical stability

    Returns:
        wY (ndarray): Whitened signal
        Z (ndarray): Whitening matrix
    """
    n_channels, n_samples = Y.shape
    use_svd = backend == 'svd'

    if method == 'Cholesky':
        covariance = Y @ Y.T / n_samples
        R = np.linalg.cholesky(covariance)
        Z = np.linalg.inv(R.T)
        wY = Z @ Y
        return wY, Z

    # Use SVD
    if use_svd:
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        if regularization == 'auto':
            reg = np.mean(S[len(S)//2:]**2)
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1. / np.sqrt(S**2 + reg + eps)

        if method == 'ZCA':
            Z = U @ np.diag(S_inv) @ U.T
        elif method == 'PCA':
            Z = np.diag(S_inv) @ U.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    # Use EIG
    else:
        covariance = Y @ Y.T / n_samples
        S, V = np.linalg.eigh(covariance)

        if regularization == 'auto':
            reg = np.mean(S[:len(S)//2])
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1. / np.sqrt(S + reg + eps)

        if method == 'ZCA':
            Z = V @ np.diag(S_inv) @ V.T
        elif method == 'PCA':
            Z = np.diag(S_inv) @ V.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    return wY, Z

def est_spike_times(sig, fsamp, cluster = 'kmeans', a = 2):
    """
    Estimate spike indices given a motor unit source signal and compute
    a silhouette-like metric for source quality quantification

    Parameters:
        sig (np.ndarray): Input signal (motor unit source)
        fsamp (float): Sampling rate in Hz
        cluster (string): Clustering method used to identify the spike indices
        a (float): Exponent of assymetric power law 

    Returns:
        est_spikes (np.ndarray): Estimated spike indices
        sil (float): Silhouette-like score (0 = poor, 1 = strong separation)
    """
    sig = np.asarray(sig)

    # Assymetric power law that can be useful for contrast enhancement
    sig = np.sign(sig) * sig**a

    if cluster == 'kmeans':
    
        # Detect peaks with minimum distance of 10 ms
        min_peak_dist = int(round(fsamp * 0.01))
        peaks, _ = find_peaks(sig, distance=min_peak_dist)

        if len(peaks) == 0:
            return np.array([])

        # Get peak values
        peak_vals = sig[peaks].reshape(-1, 1)

        # K-means clustering to separate signal vs. noise
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(peak_vals)
        centroids = kmeans.cluster_centers_.flatten()

        # Spikes are those in the cluster with the higher mean
        spike_cluster = np.argmax(centroids)
        est_spikes = peaks[labels == spike_cluster]

        # Compute within- and between-cluster distances
        D = kmeans.transform(peak_vals)  # Distances to both centroids
        sumd = np.sum(D[labels == spike_cluster, spike_cluster])
        between = np.sum(D[labels == spike_cluster, 1 - spike_cluster])
        
        # Silhouette-inspired score
        denom = max(sumd, between)
        sil = (between - sumd) / denom if denom > 0 else 0.0

    return est_spikes, sil

def gram_schmidt(w, B):
    """
    Stabilized Gram-Schmidt orthogonalization.

    Parameters:
        w (np.ndarray): Input vector to be orthogonalized (shape: [n,])
        B (np.ndarray): Matrix of orthogonal basis vectors in columns (shape: [n, k])

    Returns:
        u (np.ndarray): Orthogonalized vector
    """
    w = np.asarray(w, dtype=float)
    B = np.asarray(B, dtype=float)

    # Remove zero columns from B
    non_zero_cols = ~np.all(B == 0, axis=0)
    B = B[:, non_zero_cols]

    u = w.copy()
    for i in range(B.shape[1]):
        a = B[:, i]
        projection = (np.dot(u, a) / np.dot(a, a)) * a
        u = u - projection

    return u

def remove_duplicates(sources, spikes, sil, fsamp, max_shift=0.1, tol=0.001, threshold=0.3, min_num_spikes = 10):


    n_source = sources.shape[0]
    new_labels = np.arange(n_source)

    for i in np.arange(n_source):

        # Check if the source has already been labeled and if it contains enough spikes
        if new_labels[i] < i:
            continue
        elif len(spikes) < min_num_spikes:
            new_labels[i] = np.nan
            continue

        # Make binary spike train of source i    
        st1 = get_bin_spikes(spikes[i], sources.shape[1])
        
        for j in np.arange(i+1, n_source):
            # Make binary spike train of source j
            st2 = get_bin_spikes(spikes[j], sources.shape[1])
            # Compute the delay between source i and j
            corr , shift = max_xcorr(st1, st2, max_shift=int(max_shift*fsamp))
            # Compute the number of common spikes
            tp, _, _ = match_spikes(spikes[i], spikes[j], shift=shift, tol=tol*fsamp) 
            # Calculate the metaching rate and compare with threshold
            denom = max(len(spikes[i]), len(spikes[j]))
            match_score = tp / denom if denom > 0 else 0

            if match_score >= threshold:
                new_labels[j] = i

    # Get the number of unqiue sources and initalize output variables
    unique_labels = np.unique(new_labels[np.isfinite(new_labels)])
    new_sources  = np.zeros((len(unique_labels),sources.shape[1]))
    new_spikes = {i: [] for i in range(len(unique_labels))}
    new_sil = np.zeros(len(new_labels))

    # 
    for i in np.arange(len(unique_labels)):
        idx = (new_labels == unique_labels[i]).astype(int)
        best_idx = np.argmax(idx * sil)
        new_sources[i,:] = sources[best_idx,:]
        new_spikes[i] = spikes[best_idx]
        new_sil[i] = sil[best_idx]

    return new_sources, new_spikes, new_sil

def spike_triggered_average(sig, spikes, win=0.02, fsamp = 2048):
    '''
    Calculate the spike triggered average given the spike times of a source

    Parameters:
        sig (2D np.array): signal [channels x time]
        spikes (1D array): Spike indices
        fsamp (float): Sampling frequency in Hz
        win (float): Window size in seconds for MUAP template (in seconds)

    Returns:
        waveform (2D np.array): Estimated impulse response of a given source
    
    '''

    width = int(win*fsamp)
    waveform = np.zeros((sig.shape[0], 2*width+1))

    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]

    for i in np.arange(len(spikes)):
        waveform = waveform + sig[:,(spikes[i]-width):(spikes[i]+width+1)]

    waveform = waveform / len(spikes)    

    return waveform

def peel_off(sig, spikes, win=0.02, fsamp=2048):
    """
    Peel off signal component based on spike triggered average.

    Parameters:
        sig (2D np.array): signal [channels x time]
        spikes (1D array): Spike indices
        fsamp (float): Sampling frequency in Hz
        win (float): Window size in seconds for MUAP template (in seconds)

    Returns:
        residual_sig (2D np.array): Residual signal after removing component
        comp_sig (2D np.array): Estimated contribution of the given source
    """

    waveform = spike_triggered_average(sig,spikes,win,fsamp)

    width = int(win*fsamp)
    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]
    firings = np.zeros(sig.shape[1])
    firings[spikes] = 1

    comp_sig = np.zeros_like(sig)

    for i in np.arange(sig.shape[0]):
        comp_sig[i,:] = convolve(firings, waveform[i,:], 'same') 

    residual_sig = sig - comp_sig

    return residual_sig, comp_sig


def spike_dict_to_long_df(spike_dict, sort=True, fsamp = 2048):
    """
    Convert a dictionary of spike instances into a long-formatted DataFrame.

    Parameters:
        spike_dict (dict): Keys are unit IDs, values are lists or arrays of spike times.
        sort (bool): Whether to sort the result by unit and spike time.

    Returns:
        pd.DataFrame: Long-formatted DataFrame with columns ['unit_id', 'spike_time']
    """
    rows = []
    for unit_id, spikes in spike_dict.items():
        for t in spikes:
            rows.append({"source_id": unit_id, "spike_time": t/fsamp})
    
    df = pd.DataFrame(rows)
    if sort:
        df = df.sort_values(by=["source_id", "spike_time"]).reset_index(drop=True)
    return df
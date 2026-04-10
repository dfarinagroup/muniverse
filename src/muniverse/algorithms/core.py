import numpy as np
import pandas as pd
from typing import List, Literal, Optional
from scipy.fft import rfft, irfft, rfftfreq
from scipy.linalg import toeplitz
from scipy.signal import butter, filtfilt, find_peaks, firwin2, iirnotch
from scipy.stats import zscore
from sklearn.cluster import KMeans

from ..evaluation.evaluate import *


def bandpass_signals(data: np.ndarray, # (n_channels x n_samples)
                     fsamp: float, 
                     high_pass: float = 20, 
                     low_pass: float = 500, 
                     method: Literal["butter", "firwin2"] = "butter",
                     order: int | None = 2, 
                     numtabs: int | None = 101,
) -> np.ndarray:
    """
    Bandpass filter timeseries data using a digital infinite 
    impulse  response filter ("butter") or finite impulse 
    response filter ("firwin2").

    Args
    ----
        data : np.ndarray
            Input data (n_channels x n_samples)
        fsamp : float 
            Sampling frequency in Hz
        high_pass : float, default 20 
            Cut-off frequency for the high-pass filter in Hz    
        low_pass : float, default 500 
            Cut-off frequency for the low-pass filter in Hz
        method :  {"butter", "firwin2"}, default "butter"
            Filter type
        order : int | None, default 2 
            Order of the filter (required for "butter") 
        numtabs : int | None, default 101 
            Number of filter tabs (required for firwin2)

    Returns
    -------
        data : np.ndarray  
            filtered data (n_channels x n_samples)

    """

    if high_pass >= low_pass:
        raise ValueError(
            "The value of low_pass must be larger than your high_pass value."
        )

    if method == "butter":
        if order is None:
            raise ValueError(
                "If method is *butter*, order must be an integer."
            )
        b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
        data = filtfilt(b, a, data, axis=1)
    elif method == "firwin2":
        if numtabs is None:
            raise ValueError(
                "If method is *firwin2*, numtabs must be an integer."
            )
        # Normalize frequencies to Nyquist (0..1)
        nyq = fsamp / 2
        f = [0, high_pass*0.9, high_pass, low_pass, low_pass*1.1, nyq]  # small transition bands
        m = [0, 0, 1, 1, 0, 0]  # 0 outside band, 1 inside
        # Design FIR filter
        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        data = filtfilt(fir_coeff, [1.0], data, axis=1)
    else:
        raise ValueError(
            f"The specified filter type option *{method}* is invalid"
            "Must be one of *butter* or *firwin2*"
        )

    return data


def highpass_signals(
    data: np.ndarray,  # (n_channels x n_samples)
    fsamp: float,
    high_pass: float = 20,
    method: Literal["butter", "firwin2"] = "butter",
    order: int | None = 2,
    numtabs: int | None = 101,
) -> np.ndarray:
    """
    High-pass filter timeseries data using a digital infinite 
    impulse  response filter ("butter") or finite impulse 
    response filter ("firwin2").

    Args
    ----
        data : np.ndarray
            Input data (n_channels x n_samples)
        fsamp : float 
            Sampling frequency in Hz
        high_pass : float, default 20 
            Cut-off frequency for the high-pass filter in Hz    
        method :  {"butter", "firwin2"}, default "butter"
            Filter type
        order : int | None, default 2 
            Order of the filter (required for "butter") 
        numtabs : int | None, default 101 
            Number of filter tabs (required for firwin2)

    Returns
    -------
        data : np.ndarray  
            filtered data (n_channels x n_samples)

    """

    if high_pass <= 0:
        raise ValueError("high_pass must be > 0")

    if method == "butter":
        if order is None:
            raise ValueError("order must be provided for butter")

        b, a = butter(order, high_pass, fs=fsamp, btype="highpass")
        data = filtfilt(b, a, data, axis=1)

    elif method == "firwin2":
        if numtabs is None:
            raise ValueError("numtabs must be provided for firwin2")

        nyq = fsamp / 2

        f = [0, high_pass * 0.9, high_pass, nyq]
        m = [0, 0, 1, 1]

        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        data = filtfilt(fir_coeff, [1.0], data, axis=1)

    else:
        raise ValueError("method must be 'butter' or 'firwin2'")

    return data

def lowpass_signals(
    data: np.ndarray,  # (n_channels x n_samples)
    fsamp: float,
    low_pass: float = 500,
    method: Literal["butter", "firwin2"] = "butter",
    order: int | None = 2,
    numtabs: int | None = 101,
) -> np.ndarray:
    """
    Low-pass filter timeseries data using a digital infinite 
    impulse  response filter ("butter") or finite impulse 
    response filter ("firwin2").

    Args
    ----
        data : np.ndarray
            Input data (n_channels x n_samples)
        fsamp : float 
            Sampling frequency in Hz  
        low_pass : float, default 500 
            Cut-off frequency for the low-pass filter in Hz
        method :  {"butter", "firwin2"}, default "butter"
            Filter type
        order : int | None, default 2 
            Order of the filter (required for "butter") 
        numtabs : int | None, default 101 
            Number of filter tabs (required for firwin2)

    Returns
    -------
        data : np.ndarray  
            filtered data (n_channels x n_samples)

    """

    nyq = fsamp / 2

    if low_pass <= 0 or low_pass >= nyq:
        raise ValueError("low_pass must be between 0 and Nyquist frequency")

    if method == "butter":
        if order is None:
            raise ValueError("order must be provided for butter")

        b, a = butter(order, low_pass, fs=fsamp, btype="lowpass")
        data = filtfilt(b, a, data, axis=1)

    elif method == "firwin2":
        if numtabs is None:
            raise ValueError("numtabs must be provided for firwin2")

        f = [0, low_pass, low_pass * 1.1, nyq]
        m = [1, 1, 0, 0]

        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        data = filtfilt(fir_coeff, [1.0], data, axis=1)

    else:
        raise ValueError("method must be 'butter' or 'firwin2'")

    return data

def notch_signals(data: np.ndarray, 
                  fsamp: float, 
                  freqs: List[float] = [50, 100, 150], 
                  method: Literal[
                      "butter", "iirnotch", "fft_nulling", "fft_interpolation"
                  ] = "butter", 
                  order: int | None = 2, 
                  dfreq: int | None = 1,
    ) -> np.ndarray:
    """
    Notch filter (stop band) time series data using either a infinite impulse 
    response filter ("butter"), a finite impulse response filter ("iirnotch") or 
    performing filtering in the frequency domain ("fft_nulling" and "fft_interpolation"). 
    For "fft_nulling" the spectrum in the specified frequency band is set to zero, 
    for "fft_interpolation" the spectral amplitude is interpolated through by the 
    neighbourhood. Time series data is then recovered through an inverse fft.

    Args
    ----
        data : np.ndarray) 
            Input data (n_channels x n_samples)
        fsamp : float 
            Sampling frequency in Hz
        freqs : list of float, default [50, 100, 150] 
            List of frequencies to be notch filtered
        method : {"butter", "iirnotch", "fft_nulling", "fft_interpolation"}, default "butter"
            Filter type 
        order : int or None, default 2
            Order of the filter (if method is butter)
        dfreq : float or None, default 1 
            Width of the notch filter (in both directions) in Hz 
            (if method is iirnotch, fft_nulling, fft_interpolation).

    Returns
    -------
        data : np.ndarray 
            Filtered data (n_channels x n_samples)
    """

    if isinstance(freqs, float) or isinstance(freqs, int):
        freqs = [freqs]

    if method == "butter":

        for f0 in freqs:
            b, a = butter(
                order,
                [f0 - dfreq, f0 + dfreq],
                fs=fsamp,
                btype="bandstop",
            )
            data = filtfilt(b, a, data, axis=1)

    elif method == "iirnotch":
        for f0 in freqs:
            b, a = iirnotch(f0, f0/(2*dfreq), fsamp)
            data = filtfilt(b, a, data, axis=1)        

    elif method == "fft_nulling":
        N = data.shape[1]

        spectrum = rfft(data, axis=1)
        fft_freqs = rfftfreq(N, d=1/fsamp)

        for f0 in freqs:

            # Create notch mask (1D)
            mask = np.abs(fft_freqs - f0) <= dfreq
    
            # Broadcast mask across channels
            spectrum[:, mask] = 0

        data = irfft(spectrum, n=N, axis=1)

    elif method == "fft_interpolation":
        N = data.shape[1]

        spectrum = rfft(data, axis=1)
        fft_freqs = rfftfreq(N, d=1/fsamp)

        eps = 1e-12  # avoid log(0)

        for f0 in freqs:

            mask = np.abs(fft_freqs - f0) <= dfreq
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            left = idx[0] - 1
            right = idx[-1] + 1

            # Handle edge cases
            if left < 0 or right >= len(fft_freqs):
                continue

            # magnitude and phase ---
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            # log-magnitude
            log_mag = np.log(mag + eps)

            # values at boundaries
            left_log = log_mag[:, left]   
            right_log = log_mag[:, right]

            # interpolate in log space
            interp_log = np.linspace(
                left_log,
                right_log,
                len(idx),
                axis=1
            )
            # transform to back to magnitude space
            interp_mag = np.exp(interp_log)

            # use average phase from edges (avoids phase jumps)
            left_phase = phase[:, left]
            right_phase = phase[:, right]

            # unwrap phase to avoid discontinuities
            phase_pair = np.stack([left_phase, right_phase], axis=1)
            phase_pair = np.unwrap(phase_pair, axis=1)

            interp_phase = np.linspace(
                phase_pair[:, 0],
                phase_pair[:, 1],
                len(idx),
                axis=1
            )

            # reconstruct spectrum
            spectrum[:, idx] = interp_mag * np.exp(1j * interp_phase)

        data = irfft(spectrum, n=N, axis=1)

    else:
        raise ValueError(
            f"The specified filter type option {method} is invalid"
            "Valid options are *butter*, *iirnotch*, *fft_nulling* or *fft_interpolation*"                      
        )

    return data

def find_outliers(x: np.ndarray, # (n_features, )
                  threshold: float = 3, 
                  max_iter: int = 3, 
                  tail: Literal[-1,0,1] = 0,
                  mask: np.ndarray | None = None
):
    """
    Detect ouliers by comparing the z-score of variable x against
    some threshold. This is repeaded until there are no outliers or
    the maximum number of iterations is reached. 

    Args
    ----
        x : np.ndarray (n_features, )
            Variable to test for outliers
        threshold : float, default 3 
            Threshold for outlier detection
        max_iter: int , default 3
             Maximum number of iterations
        tail : {-1,0,1} , default 0 
            Specify weather to serach for outliers   
            on both ends (0), just on the positive (1) or just 
            the negative side (-1).
        mask : np.ndarray | None , default None
            Boolean mask to exclude channels from outlier detection
            (True: bad_channel, False: good_channel)    

    Returns
    -------
        mask : np.ndarray (n_features, )
            Boolean mask (True: outlier, False: no outlier)
        
    """

    if mask is None:
        mask = np.zeros(len(x), dtype=bool)

    iter = 0
    while iter < max_iter:
        xm = np.ma.masked_array(x, mask=mask)
        if tail == 1:
            idx = zscore(xm) > threshold
        elif tail == -1:
            idx = -zscore(xm) > threshold
        else:
            idx = np.abs(zscore(xm)) > threshold 
        mask = mask + idx
        if not np.any(idx):
            break
        else:
            iter = iter + 1 

    return mask  

def extension(Y: np.ndarray, # (n_channels x n_samples)
              R: int
) -> np.ndarray:
    """
    Extend a multi-channel signal Y by an extension factor R
    using Toeplitz matrices.

    Args
    ----
        Y : np.ndarray 
            Original signal (n_channels x n_samples)
        R : int 
            Extension factor (number of lags)

    Returns
    -------
        eY : np.ndarray 
            Extended signal (n_channels x (R * n_samples))
    """
    n_channels, n_samples = Y.shape
    eY = np.zeros((n_channels * R, n_samples))

    for i in range(n_channels):
        col = np.concatenate(([Y[i, 0]], np.zeros(R - 1)))
        row = Y[i, :]
        T = toeplitz(col, row)
        eY[i * R : (i + 1) * R, :] = T

    return eY


def whitening(Y: np.ndarray, # (n_channels x n_samples)
              method: Literal["ZCA", "PCA", "Cholesky"] = "ZCA", 
              backend: Literal["ed", "svd"] = "ed", 
              regularization: str | float | None = "auto", 
              eps: Optional[float] = 1e-10
):
    """
    Whiten data using the ZCA, PCA, or Cholesky method.

    Args
    ----
        Y : np.ndarray 
            Input signal (n_channels x n_samples)
        method : {"ZCA", "PCA", "Cholesky"}, default "ZCA" 
            Whitening method 
        backend : {"ed", "svd"}, default "ed" 
            Method used to calculate eigenvalues and eigenvectors
        regularization : {"auto", float, None}, default "auto" 
            Adds a small value to the eigenvalues for regularization.
            If "auto", the mean of the second half of the eigenvalues is used.
        eps : float 
            Small epsilon added to the eigenvalues for numerical stability

    Returns
    -------
        wY : np.ndarray) 
            Whitened signal (n_channels x n_samples)
        Z : np.ndarray 
            Whitening matrix (n_channels x n_channels)

    """
    n_channels, n_samples = Y.shape
    use_svd = backend == "svd"

    if method == "Cholesky":
        covariance = Y @ Y.T / (n_samples - 1)
        R = np.linalg.cholesky(covariance)
        Z = np.linalg.inv(R.T)
        wY = Z @ Y
        return wY, Z

    # Use SVD
    if use_svd:
        covariance = Y @ Y.T / (n_samples - 1)
        # covariance = np.cov(Y)
        U, S, _ = np.linalg.svd(covariance, full_matrices=False)
        if regularization == "auto":
            reg = np.mean(S[len(S) // 2 :] ** 2)
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1.0 / np.sqrt(S + reg + eps)

        if method == "ZCA":
            Z = U @ np.diag(S_inv) @ U.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ U.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    # Use EIG
    else:
        covariance = Y @ Y.T / (n_samples - 1)
        S, V = np.linalg.eigh(covariance)

        if regularization == "auto":
            reg = np.mean(S[: len(S) // 2])
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1.0 / np.sqrt(S + reg + eps)

        if method == "ZCA":
            Z = V @ np.diag(S_inv) @ V.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ V.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    return wY, Z


def est_spike_times(sig: np.ndarray, # (n_samples, ) 
                    fsamp: float, 
                    cluster: Literal["kmeans"] = "kmeans", 
                    a: float = 2, 
                    min_delay: float = 0.01
):
    """
    Estimate spike indices given a spiky source signal and compute
    a silhouette-like metric for source quality quantification.
    To do so, (i) a asymetric power law is applied to the signal,
    (ii) a peak detection method identifies spike candidates that
    are (iii) clustered into true and false spikes.

    Args
    ----
        sig : np.ndarray (n_samples, )
            Spike-like input signal (predicted sources)
        fsamp : float 
            Sampling rate in Hz
        cluster : {"kmeans"}, default "kmeans"
            Clustering method used to identify the spike indices
        a : float , default 2
            Exponent of assymetric power law
        min_delay : float , default 0.01 
            Mimium distance between two spikes (used for peak detection)   

    Returns
    -------
        est_spikes : np.ndarray 
            Estimated spike indices
        sil : float 
            Silhouette-like score (0 = poor, 1 = strong separation)
    """
    sig = np.asarray(sig)

    # Assymetric power law that can be useful for contrast enhancement
    sig = np.sign(sig) * sig**a

    if cluster == "kmeans":

        # Detect peaks with minimum distance of 10 ms
        min_peak_dist = int(round(fsamp * min_delay))
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
        sumd = np.sum(
            D[labels == spike_cluster, spike_cluster] ** 2
        )  # Exponent 2 for obtaining the squared Euclidian distance
        between = np.sum(
            D[labels == spike_cluster, 1 - spike_cluster] ** 2
        )  # Exponent 2 for obtaining the squared Euclidian distance

        # Silhouette-inspired score
        denom = max(sumd, between)
        sil = (between - sumd) / denom if denom > 0 else 0.0

    return est_spikes, sil


def gram_schmidt(w: np.ndarray, # (n, )
                 B: np.ndarray  # (n, k)
):
    """
    Stabilized Gram-Schmidt orthogonalization.

    Args
    ----
        w : np.ndarray (n, )
            Vector to be orthogonalized 
        B : np.ndarray (n, k)
            Matrix of basis vectors in columns 

    Returns
    -------
        u : np.ndarray (n, )
            Orthogonalized vector

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


def remove_duplicates(
    sources, spikes, sil, mu_filters, fsamp, max_shift=0.1, tol=0.001, threshold=0.3
):
    """
    Sort out source duplicates from a decomposition by clustering spike trains and
    only keeping for each unique label the source with the highest quality score

    Args:
        - sources (np.ndarray): Original sources (n_mu x n_samples)
        - spikes (dict): Original spiking instances of the motor neurons
        - sil (np.ndarray): Original source quality metric
        - mu_filters (np.ndarray): Original motor unit filters
        - fsamp (float): Sampling rate in Hz
        - max_shift (float): Maximal delay between two sources in seconds
        - tol (float): All spikes with a delay lower than tolerance (in seconds) are classified identical
        - theshold (float): Minimum fraction of common spikes to classify two sources as identical

    Returns:
        - new_sources (np.ndarray): Updated sources (n_mu x n_samples)
        - new_spikes (dict): Updated spiking instances of the motor neurons
        - new_sil (np.ndarray): Updated source quality metric
        - new_filters (np.ndarray): Updated motor unit filters


    """
    n_source = sources.shape[0]
    new_labels = np.arange(n_source)

    for i in np.arange(n_source):

        # Check if the source has already been labeled
        if new_labels[i] < i:
            continue

        # Make binary spike train of source i
        st1 = get_bin_spikes(spikes[i], sources.shape[1])

        for j in np.arange(i + 1, n_source):
            # Make binary spike train of source j
            st2 = get_bin_spikes(spikes[j], sources.shape[1])
            # Compute the delay between source i and j
            _, shift = max_xcorr(st1, st2, max_shift=int(max_shift * fsamp))
            # Compute the number of common spikes
            # tp, _, _ = match_spikes(spikes[i], spikes[j], shift=shift, tol=tol*fsamp)
            tp, _, _ = match_spike_trains(st1, st2, shift=shift, tol=tol, fsamp=fsamp)
            # Calculate the metaching rate and compare with threshold
            denom = max(len(spikes[i]), len(spikes[j]))
            match_score = tp / denom if denom > 0 else 0
            # If the match score is above the theshold update the source label
            if match_score >= threshold:
                new_labels[j] = i

    # Get the number of unqiue sources and initalize output variables
    unique_labels = np.unique(new_labels)
    new_sources = np.zeros((len(unique_labels), sources.shape[1]))
    new_spikes = {i: [] for i in range(len(unique_labels))}
    new_sil = np.zeros(len(unique_labels))
    new_filters = np.zeros((mu_filters.shape[0], len(unique_labels)))

    # For each unqiue source select the one with the highest SIL score
    for i in range(len(unique_labels)):
        idx = (new_labels == unique_labels[i]).astype(int)
        best_idx = np.argmax(idx * sil)
        new_sources[i, :] = sources[best_idx, :]
        new_spikes[i] = spikes[best_idx]
        new_sil[i] = sil[best_idx]
        new_filters[:, i] = mu_filters[:, best_idx]

    return new_sources, new_spikes, new_sil, new_filters


def remove_bad_sources(
    sources, spikes, sil, mu_filters, threshold=0.9, min_num_spikes=10
):
    """
    Reject sources with a silhoeutte score below a given threshold and that do not
    contain a minimum number of spikes.

    Args:
        - sources (np.ndarray): Original sources (n_mu x n_samples)
        - spikes (dict): Original ppiking instances of the motor neurons
        - sil (np.ndarray): Original source quality metric
        - mu_filters (np.ndarray): Original motor unit filters
        - theshold (float): Sources with a SIL score below this theshold will be rejected
        - min_num_spikes (int): Sources with less spikes will be rejected

    Returns:
        - new_sources (np.ndarray): Updated sources (n_mu x n_samples)
        - new_spikes (dict): Updated spiking instances of the motor neurons
        - new_sil (np.ndarray): Updated source quality metric
        - new_filters (np.ndarray): Updated motor unit filters

    """

    bad_source_idx = np.array([])
    new_spikes = {}
    new_spike_idx = 0
    for i in np.arange(sources.shape[0]):
        if sil[i] < threshold or len(spikes[i]) < min_num_spikes:
            bad_source_idx = np.append(bad_source_idx, i)
        else:
            new_spikes[new_spike_idx] = spikes[i]
            new_spike_idx += 1

    new_sources = np.delete(sources, bad_source_idx.astype(int), axis=0)
    new_sil = np.delete(sil, bad_source_idx.astype(int), axis=0)
    new_filters = np.delete(mu_filters, bad_source_idx.astype(int), axis=1)

    return new_sources, new_spikes, new_sil, new_filters

def spike_triggered_average(sig: np.ndarray, # (n_channels, n_samples) 
                            spikes: np.ndarray, # (n_spikes, ) 
                            win: float = 0.02, 
                            fsamp: float = 2048
):
    """
    Estimate the impulse response of a finite impulse response filters 
    given the time samples of the events.

    Args
    ----
        sig : np.ndarray (n_channels, n_samples)
            Input signal 
        spikes : np.ndarray (n_spikes, )
            Array of spike indices
        win : float , default 0.02
            Window size (in both directions) in seconds used for 
            impulse response extraction     
        fsamp : float , default 2048
            Sampling frequency in Hz

    Returns
    -------
        waveform : np.ndarray 
            Estimated impulse response of a given source

    """

    width = int(win * fsamp)
    waveform = np.zeros((sig.shape[0], 2 * width + 1))

    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]

    for i in np.arange(len(spikes)):
        waveform = waveform + sig[:, (spikes[i] - width) : (spikes[i] + width + 1)]

    waveform = waveform / len(spikes)

    return waveform


def peel_off(sig: np.ndarray, # (n_channels, n_samples) 
                            spikes: np.ndarray, # (n_spikes, ) 
                            win: float = 0.02, 
                            fsamp: float = 2048
):
    """
    Peel off the signal contribution of a source with finite impulse
    response filter given the time stamps of the impulses (spikes) 
    using spike triggered averaging. The reconstruction of the 
    component signal is achieved in the frequency domain (fft/ifft). 

    Args
    ----
        sig : np.ndarray (n_channels, n_samples)
            signal 
        spikes : np.ndarray (n_spikes, ) 
            Array of spike indices
        win : float , default 0.02
            Window size in seconds for MUAP template     
        fsamp : float , default 2048
            Sampling frequency in Hz

    Returns
    -------
        residual_sig : np.ndarray 
            Residual signal after removing component
        comp_sig : np.ndarray 
            Estimated contribution of the given source
    """

    waveform = spike_triggered_average(sig, spikes, win, fsamp)

    width = int(win * fsamp)
    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]
    firings = np.zeros(sig.shape[1])
    firings[spikes] = 1

    # Zero-pad waveform to match signal shape
    L = sig.shape[1]
    pad_len = L - waveform.shape[1]
    waveform_padded = np.pad(waveform, ((0, 0), (0, pad_len)), mode="constant")

    # FFT of firings (same for all channels)
    fft_firings = rfft(firings)

    # FFT of waveform for each channel
    fft_waveform = rfft(waveform_padded, axis=1)

    # Multiply in frequency domain (broadcasting firings to each channel)
    fft_product = fft_waveform * fft_firings

    # IFFT to get time domain component signal
    comp_sig = irfft(fft_product, n=L, axis=1)

    # Correct time shift due to FFT convolution (center of kernel)
    shift = (waveform.shape[1] - 1) // 2
    comp_sig = np.roll(comp_sig, -shift, axis=1)

    residual_sig = sig - comp_sig

    return residual_sig, comp_sig, waveform


def spike_dict_to_long_df(spike_dict: dict, 
                          fsamp: float = 2048
):
    """
    Convert a dictionary of spike instances into a long-formatted DataFrame.

    Args
    ----
        spike_dict : dict 
            Dictonary of spike times {unit_id (int): list(int)}
        fsamp : float, default 2048
            Sampling frequency in Hz

    Returns
    -------
        df : pd.DataFrame 
            Long-formatted spike table (BIDS-events style)
    """

    columns = ["onset", "duration", "sample", "unit_id", "description"]

    rows = []
    for unit_id, spikes in spike_dict.items():
        for t in spikes:
            rows.append({
                "onset": t / fsamp,
                "duration": 0,
                "sample": t,
                "unit_id": unit_id, 
                "description": "motor-unit-spike"
            })

    # If no spikes were found, create an empty DataFrame 
    if not rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)

    # Drop dublicates and sort by onset
    df = df.drop_duplicates(subset=["onset", "unit_id", "sample"])
    df = df.sort_values(by=["onset"])

    return df

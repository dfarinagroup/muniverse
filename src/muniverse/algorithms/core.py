import numpy as np
import pandas as pd
from scipy.fft import rfft, irfft, rfftfreq
from scipy.linalg import toeplitz
from scipy.signal import butter, filtfilt, find_peaks, firwin2, iirnotch
from scipy.stats import zscore
from sklearn.cluster import KMeans

from ..evaluation.evaluate import *


def bandpass_signals(emg_data: np.ndarray, 
                     fsamp: float, 
                     high_pass: float = 20, 
                     low_pass: float = 500, 
                     method: str = "butter",
                     order: int = 2, 
                     numtabs: int = 101,
) -> np.ndarray:
    """
    Bandpass filter emg data

    Args:
        emg_data (np.ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        low_pass (float): Cut-off frequency for the low-pass filter
        high_pass (float): Cut-off frequency for the high-pass filter
        ftype (str): Filter type (butter or firwin2)
        order (int): Order of the filter (butter) 
        numtabs (int): Number of filter tabs (firwin2)

    Returns:
        np.ndarray : filtered emg data (n_channels x n_samples)
    """

    if method == "butter":
        b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
        emg_data = filtfilt(b, a, emg_data, axis=1)
    elif method == "firwin2":
        # Normalize frequencies to Nyquist (0..1)
        nyq = fsamp / 2
        f = [0, high_pass*0.9, high_pass, low_pass, low_pass*1.1, nyq]  # small transition bands
        m = [0, 0, 1, 1, 0, 0]  # 0 outside band, 1 inside
        # Design FIR filter
        fir_coeff = firwin2(numtabs, f, m, fs=fsamp)
        emg_data = filtfilt(fir_coeff, [1.0], emg_data, axis=1)
    else:
        raise ValueError(
            f"The specified filter type option *{method}* is invalid"
            "Must be one of *butter* or *firwin2*"
        )

    return emg_data


def notch_signals(emg_data: np.ndarray, 
                  fsamp: float, 
                  freqs: list[float] = [50, 100, 150], 
                  method: str = "butter", 
                  order: int = 2, 
                  dfreq: float = 1,
    ) -> np.ndarray:
    """
    Notch filter emg data

    Args:
        emg_data (np.ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        freqs (list[float]): List of frequencies to be filtered
        method (str): Filter type (butter, iirnotch, fft_nulling, fft_interpolation)
        order (int): Order of the filter (if type = butter)
        dfreq (float): width of the notch filter (plus/minus dfreq) (if iirnotch, fft_nulling, fft_interpolation)

    Returns:
        np.ndarray : filtered emg data (n_channels x n_samples)
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
            emg_data = filtfilt(b, a, emg_data, axis=1)

    elif method == "iirnotch":
        for f0 in freqs:
            b, a = iirnotch(f0, f0/(2*dfreq), fsamp)
            emg_data = filtfilt(b, a, emg_data, axis=1)        

    elif method == "fft_nulling":
        N = emg_data.shape[1]

        spectrum = rfft(emg_data, axis=1)
        fft_freqs = rfftfreq(N, d=1/fsamp)

        for f0 in freqs:

            # Create notch mask (1D)
            mask = np.abs(fft_freqs - f0) <= dfreq
    
            # Broadcast mask across channels
            spectrum[:, mask] = 0

        emg_data = irfft(spectrum, n=N, axis=1)

    elif method == "fft_interpolation":
        N = emg_data.shape[1]

        spectrum = rfft(emg_data, axis=1)
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

        emg_data = irfft(spectrum, n=N, axis=1)

    else:
        raise ValueError(
            f"The specified filter type option {method} is invalid"
            "Valid options are *butter*, *iirnotch*, *fft_nulling* or *fft_interpolation*"                      
        )

    return emg_data

def find_outliers(x, threshold=3, max_iter=3, tail=0):
    """
    Detect ouliers by comparing the z-score of variable x against
    some threshold. This is repeaded until there are no outliers or
    the maximum number of iterations is reached. 

    Args:
        x (np.array): Variable to test for outliers
        threshold (float): Threshold for outlier detection
        max_iter (int): Maximum number of iterations
        tail {-1,0,1}: Specify weather to serach for outliers   
            on both ends (0), just on the positive (1) or just 
            the negative side (-1).

    Return:
        bad_idx (np.array): List of bad channels (integer index) 
        
    """

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

    bad_idx = np.where(mask)[0]    

    return bad_idx  


def reject_bad_channels(data, bad_channels):
    """
    Reject a list of bad channels from the data matrix

    Args:
        data (ndarray): Data matrix (channels x samples)
        bad_channels (ndarray): List of bad channel indices

    Returns:
        data (ndarray): Updated data matrix 
        mask (ndarray): Array showing if channels are used (True) or rejected (False)
    """

    mask = np.ones(data.shape[1], dtype=bool)
    mask[bad_channels] = False
    data = data[mask,:]

    return data, mask


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
        eY[i * R : (i + 1) * R, :] = T

    return eY


def whitening(Y, method="ZCA", backend="ed", regularization="auto", eps=1e-10):
    """
    Adaptive whitening function using ZCA, PCA, or Cholesky.

    Parameters:
        Y (ndarray): Input signal (n_channels x n_samples)
        method (str): Whitening method: 'ZCA', 'PCA', 'Cholesky'
        backend (str): 'ed', or 'svd'
        regularization (str or float): 'auto', float value, or None
        eps (float): Small epsilon for numerical stability

    Returns:
        wY (ndarray): Whitened signal
        Z (ndarray): Whitening matrix
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


def est_spike_times(sig, fsamp, cluster="kmeans", a=2, min_delay=0.01):
    """
    Estimate spike indices given a motor unit source signal and compute
    a silhouette-like metric for source quality quantification

    Args:
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


def gram_schmidt(w, B):
    """
    Stabilized Gram-Schmidt orthogonalization.

    Args:
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

def spike_triggered_average(sig, spikes, win=0.02, fsamp=2048):
    """
    Calculate the spike triggered average given the spike times of a source

    Parameters:
        sig (2D np.array): signal [channels x time]
        spikes (1D array): Spike indices
        fsamp (float): Sampling frequency in Hz
        win (float): Window size in seconds for MUAP template (in seconds)

    Returns:
        waveform (2D np.array): Estimated impulse response of a given source

    """

    width = int(win * fsamp)
    waveform = np.zeros((sig.shape[0], 2 * width + 1))

    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]

    for i in np.arange(len(spikes)):
        waveform = waveform + sig[:, (spikes[i] - width) : (spikes[i] + width + 1)]

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


def spike_dict_to_long_df(spike_dict, fsamp=2048):
    """
    Convert a dictionary of spike instances into a long-formatted DataFrame.

    Parameters:
        spike_dict (dict): Key: unit_id; Value: array of spike indices
        fsamp (float): Sampling frequency to convert sample indices to time.

    Returns:
        pd.DataFrame: Long-formatted DataFrame (BIDS-events style)
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

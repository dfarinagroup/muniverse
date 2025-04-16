import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import toeplitz

def bandpass_signals(emg_data, fsamp, low_pass = 20, high_pass = 500, order = 2):
    """
    Bandpass filter emg data using a butterworth filter

    Args:
        emg_data (ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        low_pass (float): Cut-off frequency for the low-pass filter
        high_pass (float): Cut-off frequency for the high-pass filter
        order (int): Order of the filter

    Returns:
        ndarray : filtered emg data (n_channels x n_samples)
    """

    b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype='band')
    emg_data = filtfilt(b,a,emg_data, axis=1)

    return emg_data

def notch_signals(emg_data, fsamp, nfreq = 50, dfreq = 1, order = 2):
    """
    Notch filter emg data using a butterworth filter

    Args:
        emg_data (ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        nfreq (float): frequency to be filtered
        dfreq (float): width of the notch filter (plus/minus dfreq)
        order (int): Order of the filter

    Returns:
        ndarray : filtered emg data (n_channels x n_samples)
    """

    b, a = butter(order, [nfreq-dfreq, nfreq+dfreq], fs=fsamp, btype='bandstop')
    emg_data = filtfilt(b,a,emg_data, axis=1)

    return emg_data

def extension(SIG, R):
    """
    Extend a multi-channel signal Y by an extension factor R
    using Toeplitz matrices and subtract the mean from each extended channel.

    Parameters:
        SIG (ndarray): Original signal (n_channels x n_samples)
        R (int): Extension factor (number of lags)

    Returns:
        eSIG (ndarray): Extended signal (n_channels * R x n_samples), zero-mean per row
    """
    n_channels, n_samples = SIG.shape
    eSIG = np.zeros((n_channels * R, n_samples))

    for i in range(n_channels):
        col = np.concatenate(([SIG[i, 0]], np.zeros(R - 1)))
        row = SIG[i, :]
        T = toeplitz(col, row)
        eSIG[i * R:(i + 1) * R, :] = T

    # Subtract mean from each extended row (channel)
    eSIG -= np.mean(eSIG, axis=1, keepdims=True)

    return eSIG

def whitening(eSIG, method='ZCA', backend='eig', regularization='auto', eps=1e-10):
    """
    Adaptive whitening function using ZCA, PCA, or Cholesky.

    Parameters:
        eSIG (ndarray): Input signal (n_channels x n_samples)
        method (str): Whitening method: 'ZCA', 'PCA', 'Cholesky'
        backend (str): 'eig', or 'svd'
        regularization (str or float): 'auto', float value, or None
        eps (float): Small epsilon for numerical stability

    Returns:
        wSIG (ndarray): Whitened signal
        whitening_matrix (ndarray): Whitening matrix
    """
    n_channels, n_samples = eSIG.shape
    use_svd = backend == 'svd'

    if method == 'Cholesky':
        covariance = eSIG @ eSIG.T / n_samples
        R = np.linalg.cholesky(covariance)
        whitening_matrix = np.linalg.inv(R.T)
        wSIG = whitening_matrix @ eSIG
        return wSIG, whitening_matrix

    # Use SVD
    if use_svd:
        U, S, Vt = np.linalg.svd(eSIG, full_matrices=False)
        if regularization == 'auto':
            reg = np.mean(S[len(S)//2:]**2)
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1. / np.sqrt(S**2 + reg + eps)

        if method == 'ZCA':
            whitening_matrix = U @ np.diag(S_inv) @ U.T
        elif method == 'PCA':
            whitening_matrix = np.diag(S_inv) @ U.T
        else:
            raise ValueError("Unknown method.")
        wSIG = whitening_matrix @ eSIG

    # Use EIG
    else:
        covariance = eSIG @ eSIG.T / n_samples
        S, V = np.linalg.eigh(covariance)

        if regularization == 'auto':
            reg = np.mean(S[:len(S)//2])
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1. / np.sqrt(S + reg + eps)

        if method == 'ZCA':
            whitening_matrix = V @ np.diag(S_inv) @ V.T
        elif method == 'PCA':
            whitening_matrix = np.diag(S_inv) @ V.T
        else:
            raise ValueError("Unknown method.")
        wSIG = whitening_matrix @ eSIG

    return wSIG, whitening_matrix
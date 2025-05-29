import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_signals(emg_data, fsamp, high_pass=20, low_pass=500, order=2):
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

    b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
    emg_data = filtfilt(b, a, emg_data, axis=1)

    return emg_data


def notch_signals(emg_data, fsamp, nfreq=50, dfreq=1, order=2, n_harmonics=3):
    """
    Notch filter emg data using a butterworth filter

    Args:
        emg_data (ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        nfreq (float): frequency to be filtered
        dfreq (float): width of the notch filter (plus/minus dfreq)
        order (int): Order of the filter
        n_harmonics: Number of harmonics to be filtered

    Returns:
        ndarray : filtered emg data (n_channels x n_samples)
    """

    harmonics = nfreq * np.arange(1, n_harmonics + 1)

    for i in np.arange(n_harmonics):
        b, a = butter(
            order,
            [harmonics[i] - dfreq, harmonics[i] + dfreq],
            fs=fsamp,
            btype="bandstop",
        )
        emg_data = filtfilt(b, a, emg_data, axis=1)

    return emg_data

import numpy as np
import pandas as pd
from scipy.signal import convolve, correlate, correlation_lags, find_peaks
from scipy.stats import kurtosis, skew
from scipy.optimize import linear_sum_assignment

def match_spikes(
        s1, 
        s2, 
        shift=0, 
        tol=0.001
):
    """
    Match spike times of two neurons given time shift and tolerance.

    Args
    ----
        s1 : np.ndarray 
            Spike times of the first neuron (in seconds)
        s2 : np.ndarray 
            Spike times of the second neuron (in seconds)
        shift : int 
            Delay between the spike trains (in seconds)
        tol : float 
            Common spikes are in a window [spike-tol, spike+tol] 

    Returns
    -------
        tp : int 
            Number of true positive spikes
        fp : int 
            Number of false positive spikes
        fn : int 
            Number of false negative spikes

    """

    t1 = np.sort(s1)
    t2 = np.sort(s2 + shift)

    matched_1 = np.zeros(len(t1), dtype=bool)
    matched_2 = np.zeros(len(t2), dtype=bool)

    i, j = 0, 0
    while i < len(t1) and j < len(t2):
        dt = t1[i] - t2[j]
        if abs(dt) <= tol:
            matched_1[i] = True
            matched_2[j] = True
            i += 1
            j += 1
        elif dt < -tol:
            i += 1
        else:
            j += 1

    tp = matched_1.sum()
    fp = (~matched_1).sum()
    fn = (~matched_2).sum()
    return tp, fp, fn


def match_spike_trains(
        s1: np.ndarray, 
        s2: np.ndarray, 
        shift: float = 0, 
        tol: float = 0.001, 
        fsamp: float = 10000
) -> tuple[int, int, int]:
    """
    Match spike trains of two neurons given sample shift and tolerance.

    Args
    ----
        s1 : np.ndarray 
            Binary spike train of the first neuron
        s2 : np.ndarray 
            Binary spike train of the second neuron
        shift : int 
            Delay between the spike trains (in samples)
        tol : float 
            Common spikes are in a window [spike-tol, spike+tol] (unit: seconds)
        fsamp  : float 
            Sampling frequency in Hz

    Returns
    -------
        tp : int 
            Number of true positive spikes
        fp : int 
            Number of false positive spikes
        fn : int 
            Number of false negative spikes

    """

    s2 = np.roll(s2, shift)

    kernel = np.ones(int(2 * tol * fsamp) + 1)

    masked_s1 = convolve(s1, kernel, mode="same") > 0
    masked_s2 = convolve(s2, kernel, mode="same") > 0

    tp = np.logical_and(s1 > 0, masked_s2).sum()
    fp = np.logical_and(s1 > 0, ~masked_s2).sum()
    fn = np.logical_and(s2 > 0, ~masked_s1).sum()

    return tp, fp, fn


def get_bin_spikes(
        spike_indices: np.ndarray, 
        n_samples: int
) -> np.ndarray:
    """
    Make binary spike trains given a set of spike indices

    Args
    ----
        spike_indices : np.ndarray of int 
            Array of spike indices 
        n_samples : int 
            Number of time samples

    Returns
    -------
        spike_train : ndarray 
            Binary spike train vector

    """

    spike_train = np.zeros(n_samples, dtype=int)
    spike_train[spike_indices] = 1

    return spike_train


def bin_spikes(
        spike_times: np.ndarray, 
        fsamp: float = 2048, 
        t_start: float = 0, 
        t_end: float = 60
) -> np.ndarray:
    """
    Make binary spike trains given a set of spike times

    Args:
        spike_times : np.ndarray 
            Array of spike times (in seconds)
        fsamp : float 
            Sampling rate in Hz 
        t_start : float 
            Start of the time window to be considered (in seconds)
        t_end : float 
            End of the time window to be considered (in seconds)

    Returns:
        spike_train : np.ndarray 
            Binary spike train vector

    """

    n_samples = int(np.ceil((t_end - t_start) * fsamp)) + 1
    spike_train = np.zeros(n_samples, dtype=int)
    spike_indices = np.round(
        (spike_times - t_start) * fsamp
    ).astype(int)  
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < n_samples)]
    spike_train[spike_indices] = 1

    return spike_train


def best_time_shift(
    spikes1, spikes2, tolerance=0.001, max_shift=0.01, shift_step=0.0005
):
    """ Try multiple time shifts and return the one with maximum TP """

    best_tp = 0
    best_shift = 0.0
    best_fp, best_fn = 0, 0

    shifts = np.arange(-max_shift, max_shift + shift_step, shift_step)
    for shift in shifts:
        tp, fp, fn = match_spikes(spikes1, spikes2, shift, tolerance)
        if tp > best_tp:
            best_tp, best_fp, best_fn = tp, fp, fn
            best_shift = shift

    return best_tp, best_fp, best_fn, best_shift


def max_xcorr(
        sig1: np.ndarray, 
        sig2: np.ndarray, 
        max_shift: int = 1000
) -> tuple[float, int]:
    """
    Align two signals by finding the delay maximizing 
    their cross-correlation.

    Args
    ----
        sig1 : np.ndarray
            Reference signal
        sig2 : np.ndarray 
            Another signal
        max_shift : int 
            Maximum delay (in samples) between the two signals

    Returns
    -------
        overlap : float 
            Maximum cross-correlation
        best_shift : int 
            Delay that maximizes the cross-correlation

    """

    # corr = correlate(sig1, sig2, mode="full")
    # lags = correlation_lags(len(sig1), len(sig2), mode="full")
    # mask = (lags >= -max_shift) & (lags <= max_shift)
    # corr_win = corr[mask]
    # lags_win = lags[mask]
    # best_idx = np.argmax(corr_win)
    # overlap = corr_win[best_idx]
    # best_shift = lags_win[best_idx]

    corr = correlate(sig1, sig2, mode="full", method="fft")

    center = len(sig2) - 1

    start = max(0, center - max_shift)
    stop = min(len(corr), center + max_shift + 1)

    corr_win = corr[start:stop]

    best_idx = np.argmax(corr_win)

    overlap = corr_win[best_idx]
    best_shift = (start + best_idx) - center

    return overlap, best_shift


def label_sources(
        df: pd.DataFrame, 
        fsamp: float, 
        t_start: float = 0, 
        t_end: float = -1, 
        threshold: float = 0.3, 
        max_shift: float = 0.1, 
        tol: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find common sources given a set of spike trains

    Args
    ----
        df : pd.DataFrame 
            Data Frame containing spiking neuron activities 
        fsamp : float 
            Sampling frequecny in Hz
        t_start : float 
            Start of the time window to be considered (in seconds)
        t_end : float 
            End of the time window to be considered (in seconds)
        theshold : float 
            Common sources need to have a matching score higher than the theshold
        max_shift :float 
            Maximum delay between two sources (in seconds)
        tol : float 
            Common spikes need to be in the window [spike-tol, spike+tol]


    Returns
    -------
        labels : np.ndarray 
            new labels of the sources
        match_matrix : np.ndarray 
            matching scores between all pairs of sources

    """

    if t_end == -1:
        t_end = df["onset"].max() + 0.1

    units = sorted(df["unit_id"].unique())
    n_source = len(units)
    labels = np.arange(n_source)
    match_matrix = np.identity(n_source)

    for i in np.arange(n_source):
        spikes_1 = df[df["unit_id"] == units[i]]["onset"].values
        spikes_1 = spikes_1[(spikes_1 >= t_start) & (spikes_1 < t_end)]
        st1 = bin_spikes(spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end)

        for j in np.arange(i + 1, n_source):
            spikes_2 = df[df["unit_id"] == units[j]]["onset"].values
            spikes_2 = spikes_2[(spikes_2 >= t_start) & (spikes_2 < t_end)]
            st2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
            _, shift = max_xcorr(st1, st2, max_shift=int(max_shift * fsamp))
            tp, _, _ = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol)
            # tp, _, _ = match_spike_trains(
            #         st1, st2, shift=shift, tol=tol, fsamp=fsamp
            #     )
            denom = max(len(spikes_1), len(spikes_2))
            match_score = tp / denom if denom > 0 else 0

            match_matrix[i, j] = match_score
            match_matrix[j, i] = match_score

            if match_score >= threshold:
                labels[j] = labels[i]

    return labels, match_matrix


def signal_based_quality_metrics(
    source: np.ndarray, 
    spikes: np.ndarray, 
    fsamp: float, 
    min_peak_dist: float = 0.01, 
    match_dist: float = 0.001
) -> dict:
    """
    Compute a set of signal based quality metrics

    Args:
        source : np.ndarray 
            The predicted source
        spikes : np.ndarray 
            Indices of the predicted spikes
        fsamp : float Sampling frequency (Hz).
        min_peak_dist : float 
            Minimum distance between peaks (for peak detection) in seconds.
        match_dist : float 
            Window size (±) around predicted spikes to exclude from background.

    Returns
    -------
        quality_metrics : dict 
            Dictonary of source quality metrics

    """

    quality_metrics = {}

    # Number of Spikes
    quality_metrics["n_spikes"] = len(spikes)
    # Skewness
    quality_metrics["skew_val"] = skew(source)
    # Kurtosis
    quality_metrics["kurt_val"] = kurtosis(source)
    # Mean peak amplitude
    quality_metrics["peak_height"] = np.mean(source[spikes])
    # Coefficient of variation of the peaks
    quality_metrics["cov_peak"] = np.std(source[spikes]) / np.mean(source[spikes])

    # Silhouette-like score
    sil, background_peaks = pseudo_sil_score(
        source, spikes, fsamp, min_peak_dist=min_peak_dist, match_dist=match_dist
    )

    quality_metrics["sil"] = sil

    pred_amps = source[spikes]
    back_amps = source[background_peaks]

    # Seperability metric
    quality_metrics["sep_prctile90"] = (
        np.percentile(pred_amps, 10) - np.percentile(back_amps, 90)
    ) / np.mean(pred_amps)

    # Seperation in terms of standard deviations
    quality_metrics["sep_std"] = (np.mean(pred_amps) - np.mean(back_amps)) / np.std(
        pred_amps
    )

    # Pulse-to-noise ration (PNR)
    pnr, noise_indices = calc_pnr(source, spikes)
    quality_metrics["pnr"] = pnr

    # Z-score of the peak amplitude
    quality_metrics["z_score_height"] = np.mean(source[spikes]) / np.std(
        source[noise_indices]
    )

    return quality_metrics


def pseudo_sil_score(
        source: np.ndarray, # (n_samples, ) 
        spikes: np.ndarray, # (n_spikes, ) 
        fsamp: float, 
        min_peak_dist: float = 0.01, 
        match_dist: float = 0.001
) -> tuple[float, np.ndarray, tuple]:
    """
    Computes a silhouette-like quality score for predicted spikes based on
    peak detection and background spike amplitudes.

    Args
    ----
        source : np.ndarray 
            The predicted spiky source signal
        redicted_spikes : np.ndarray 
            Indices of predicted spike times
        fsamp : float 
            Sampling frequency in Hz
        min_peak_distance_ms : float 
            Minimum distance between peaks (for peak detection) in seconds
        match_dist : float 
            Window size (± s) around predicted spikes to exclude from background.

    Returns
    -------
        sil : float 
            Silhouette-like quality score
        background_spikes : np.ndarray 
            Indices of the background spikes
        centroids : tuple 
            Centrodis of the peak and noise cluster
    """

    spikes = np.asarray(spikes, dtype=int)
    match_window = int(round(fsamp * match_dist))
    min_dist = int(round(fsamp * min_peak_dist))

    sig = source * abs(source)

    # Step 1: Detect peaks
    detected_peaks, _ = find_peaks(sig, distance=min_dist)

    # Step 2: Exclude predicted spikes ± match_window
    mask = np.ones(len(detected_peaks), dtype=bool)
    for spike in spikes:
        mask &= np.abs(detected_peaks - spike) > match_window

    background_spikes = detected_peaks[mask]

    # Step 3: Compute amplitudes
    if len(spikes) < 2 or len(background_spikes) == 0:
        return 0.0

    pred_amps = sig[spikes].reshape(-1, 1)
    back_amps = sig[background_spikes].reshape(-1, 1)

    centroids = [np.mean(pred_amps), np.mean(back_amps)]
    within = np.sum((pred_amps - centroids[0]) ** 2) if len(pred_amps) > 1 else 0.0
    between = np.sum((pred_amps - centroids[1]) ** 2)

    sil = (between - within) / max(between, within) if max(between, within) > 0 else 0.0

    return sil, background_spikes


def calc_pnr(
        source: np.ndarray, # (n_samples, ) 
        spikes_idx: np.ndarray # (n_spikes, ) 
):
    """
    Calculate the pulse-to-noise ratio, i.e., a logarithmic measure of the amplitude of the
    spike cluster compared to the background noise in a source.

    Args
    ----
        source : np.ndarray 
            The predicted spiky source signal
        spikes_idx : np.ndarray 
            Array of spike indices predicted by a decomposition

    Returns
    -------
        pnr : float 
            Pulse-to-noise ratio of a source
        noise_indices : np.ndarray 
            Array of indices associated with noise

    """

    # Calculate PNR
    signal_length = len(source)

    sig = source * abs(source)

    expanded_indices = set()
    for idx in spikes_idx:
        for neighbor in [idx - 1, idx, idx + 1]:
            if 0 <= neighbor < signal_length:
                expanded_indices.add(neighbor)

    expanded_indices = np.array(sorted(expanded_indices))

    all_indices = np.arange(signal_length)
    noise_indices = np.setdiff1d(all_indices, expanded_indices)

    peak_cluster = sig[expanded_indices] ** 2
    noise_cluster = sig[noise_indices] ** 2

    pnr = 10 * np.log10(np.mean(peak_cluster) / np.mean(noise_cluster))

    return pnr, noise_indices


def get_basic_spike_statistics(
        spike_times: np.ndarray, 
        min_num_spikes: int = 10
) -> tuple[float, float]:
    """
    Compute the mean firing rate (Hz) and the coefficient of variation
    of the interspike intervalls given a spike train

    Args
    ----
        spike_times : np.ndarray
            Array of spike times (in seconds)
        min_num_spikes : int 
            Minimum number of spikes required for the analysis

    Returns
    -------
        cov : float 
            Coefficient of variation of the interspike intervalls
        mean_fr : float 
            Mean discharge rate of the neuron

    """

    cov = np.inf
    mean_fr = 0

    if len(spike_times) > min_num_spikes:
        # Get the interspike intervals
        isi = np.diff(spike_times)
        # Reject periods where the neuron was inactive
        std_isi = np.std(isi)
        isi = isi[isi <= 10 * std_isi]

        if len(isi) > 1:
            # Compute the CoV of the interspike intervals
            cov = np.std(isi) / np.mean(isi)
            # Compute the mean discharge rate
            mean_fr = 1 / np.mean(isi)

    return cov, mean_fr

def evaluate_spike_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    t_start: float = 0,
    t_end: float = -1,
    tol: float = 0.001,
    max_shift: float = 0.1,
    fsamp: float = 2048,
    threshold: float = 0.3,
    mask: np.ndarray | None = None,
    pre_matched: bool = False,
):
    """
    Match spiking motor unit activity between two data sets.

    Args
    ----
        df1 : pd.DataFrame
            Data Frame containing spiking neuron activities of dataset 1. 
            It must include the columns 'unit_id' and 'onset'.

        df2 : pd.DataFrame
            Data Frame containing spiking neuron activities of dataset 2.
            It must include the columns 'unit_id' and 'onset'.

        t_start : float, default 0 
            Start of the time window to be considered (in seconds)

        t_end : float, defualt 60 
            End of the time window to be considered (in seconds)

        tol : float , default 0.001
            Common spikes need to be in the window [spike-tol, spike+tol]
            in seconds

        max_shift : float 
            Maximum delay between two spike trains (in seconds)

        fsamp : float 
            Sampling rate (in Hz) of the binary spike train

        theshold : float , default 0.3
            Common sources need to have a matching score higher than the theshold

        mask : np.ndarray of bool | None , default None
            Boolean mask to indicate units to be excluded (optional) 

        pre_matched : bool , default False
            If True, it is assumed that units have already been pre-matched   

    Returns
    -------
        results : pd.DataFrame 
            Table of matched units

    """

    if "event_type" in df1.columns:
        df1 = df1[
            df1["event_type"] == "motor-unit-spike"
        ]
    if "event_type" in df2.columns:
        df2 = df2[
            df2["event_type"] == "motor-unit-spike"
        ]    

    if t_end == -1:
        t_end = max(
                df1["onset"].max(), df2["onset"].max()
            )  + 0.1    

    source_labels_1 = sorted(df1["unit_id"].unique())
    source_labels_2 = sorted(df2["unit_id"].unique())

    if mask is None:
        mask = np.ones(len(source_labels_1), dtype=bool)

    # Asign source labels using the Hungarian method
    if not pre_matched:
        cost_matrix = np.ones((len(source_labels_1), len(source_labels_2)))

        for i, l1 in enumerate(source_labels_1):

            if ~mask[i]:
                cost_matrix[i, :] = 100

            spikes_1 = df1[df1["unit_id"] == l1]["onset"].values
            spikes_1 = spikes_1[(spikes_1 >= t_start) & (spikes_1 < t_end)]
            spike_train_1 = bin_spikes(
                spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end
            )

            for j, l2 in enumerate(source_labels_2):

                spikes_2 = df2[df2["unit_id"] == l2]["onset"].values
                spikes_2 = spikes_2[(spikes_2 >= t_start) & (spikes_2 < t_end)]
                spike_train_2 = bin_spikes(
                    spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end
                )
                _, shift = max_xcorr(
                    spike_train_1, spike_train_2, max_shift=int(max_shift * fsamp)
                )
                tp, fp, fn = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol)
                # tp, fp, fn = match_spike_trains(
                #     spike_train_1, spike_train_2, shift=shift, tol=tol, fsamp=fsamp
                # )
                denom = 2*tp + fp + fn
                match_score = 2*tp / denom if denom > 0 else 0

                cost_matrix[i, j] = 1 - match_score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

    else:
        row_ind = np.arange(len(source_labels_1))
        col_ind = np.arange(len(source_labels_2))

    results = []

    for l1 in source_labels_1:
        if l1 in row_ind:
            idx = np.argmax(row_ind == l1)
            l2 = source_labels_2[col_ind[idx]]

            spikes_1 = df1[df1["unit_id"] == l1]["onset"].values
            spikes_1 = spikes_1[(spikes_1 >= t_start) & (spikes_1 < t_end)]
            spike_train_1 = bin_spikes(
                spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end
            )

            spikes_2 = df2[df2["unit_id"] == l2]["onset"].values
            spikes_2 = spikes_2[(spikes_2 >= t_start) & (spikes_2 < t_end)]
            spike_train_2 = bin_spikes(
                spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end
            )
            _, shift = max_xcorr(
                spike_train_1, spike_train_2, max_shift=int(max_shift * fsamp)
            )
            tp, fp, fn = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol)
            # tp, fp, fn = match_spike_trains(
            #     spike_train_1, spike_train_2, shift=shift, tol=tol, fsamp=fsamp
            # )
            denom = 2*tp + fp + fn
            match_score = 2*tp / denom if denom > 0 else 0
            delay = shift / fsamp
            if match_score < threshold:
                l2, tp, fp, fn, delay = None, 0, len(spikes_1), 0, None
        else:
            l2, tp, fp, fn, delay = None, 0, len(spikes_1), 0, None

        results.append(
            {
                "unit_id": l1,
                "unit_id_ref": l2,
                "delay_seconds": delay,
                "TP": tp,
                "FN": fn,
                "FP": fp,
            }
        )

    return pd.DataFrame(results)

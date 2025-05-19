import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, find_peaks, convolve
from scipy.stats import kurtosis, skew

def match_spikes(s1, s2, shift=0, tol=0.001):
    """
    Match spike times of two neurons given time shift and tolerance.

    Args:
        - s1 (ndarray): Spike times of the first neuron (in seconds)
        - s2 (ndarray): Spike times of the second neuron (in seconds) 
        - shift (float): Temporal delay between the spinking activity (in seconds)
        - tol (float): Common spikes are in a window [spike-tol, spike+tol]

    Returns:
        - tp (float): Number of true positive spikes
        - fp (float): Number of false positive spikes
        - fn (float): Number of false negative spikes  

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

def match_spike_trains(s1, s2, shift=0, tol=0.001, fsamp=10000):
    """
    Match spike times of two neurons given time shift and tolerance.

    Args:
        - s1 (ndarray): Spike times of the first neuron (in seconds)
        - s2 (ndarray): Spike times of the second neuron (in seconds) 
        - shift (float): Temporal delay between the spinking activity (in seconds)
        - tol (float): Common spikes are in a window [spike-tol, spike+tol]
        - fsamp (float): Sampling frequency in Hz

    Returns:
        - tp (float): Number of true positive spikes
        - fp (float): Number of false positive spikes
        - fn (float): Number of false negative spikes    

    """

    s2 = np.roll(s2,shift)

    kernel = np.ones(int(2 * tol * fsamp) + 1)

    masked_s1 = convolve(s1,kernel, mode='same') > 0
    masked_s2 = convolve(s2,kernel, mode='same') > 0

    tp = np.logical_and(s1>0, masked_s2).sum()
    fp = np.logical_and(s1>0, ~masked_s2).sum()
    fn = np.logical_and(s2>0, ~masked_s1).sum()

    return tp, fp, fn

def get_bin_spikes(spike_indices, n_samples):
    '''
    Make binary spike trains given a set of spike indices

    Args:
        spike_indices (ndarray): Array of spike indices (i.e, integers) 
        n_samples (int): Number of time samples

    Returns:
        spike_train (ndarray): Binary spike train vector    

    '''

    spike_train = np.zeros(n_samples, dtype=int)
    spike_train[spike_indices] = 1

    return spike_train

def bin_spikes(spike_times, fsamp = 10000, t_start=0, t_end = 60):
    '''
    Make binary spike trains given a set of spike times

    Args:
        spike_times (ndarray): Array of spike times (in seconds) 
        fsamp (float): Sampling rate in Hz of the binary spike train
        t_start (float) : Start of the time window to be considered (in seconds)
        t_end (float): End of the time window to be considered (in seconds)


    Returns:
        spike_train (ndarray): Binary spike train vector

    '''

    step_size = 1/fsamp
    n_samples = int(np.ceil((t_end - t_start) / step_size)) + 1
    spike_train = np.zeros(n_samples, dtype=int)
    spike_indices = np.round((spike_times - t_start) / step_size).astype(int) #(spike_times / step_size).astype(int)
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < n_samples)]
    spike_train[spike_indices] = 1

    return spike_train

def best_time_shift(spikes1, spikes2, tolerance=0.001, max_shift=0.01, shift_step=0.0005):
    """
    Try multiple time shifts and return the one with maximum TP.
    """
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

def max_xcorr(sig1, sig2, max_shift=1000):
    '''
    Align two spike trains by finding the delay maximizing their cross-correlation.

    Arguments:
        - a (ndarray): Reference signal
        - b (ndarray): Another signal 
        - max_shift (int): Maximum delay (in samples) between the two signals

    Outputs:
        - overlap (float): Maximum cross-correlation
        - best_shift (float): Delay that maximizes the cross-correlation     

    '''
    
    corr = correlate(sig1, sig2, mode='full')
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    mask = (lags >= -max_shift) & (lags <= max_shift)
    corr_win = corr[mask]
    lags_win = lags[mask]
    best_idx = np.argmax(corr_win)
    overlap = corr_win[best_idx]
    best_shift = lags_win[best_idx]

    #corr = np.correlate(sig1, sig2, mode='full')
    #mid = len(sig1) - 1
    #shift_range = range(mid - max_shift, mid + max_shift + 1)
    #best_shift = max(shift_range, key=lambda i: corr[i])
    #overlap = corr[best_shift]
    #best_shift = best_shift - mid
    return overlap, best_shift

def label_sources(df, fsamp=10000, t_start=0, t_end=60, threshold=0.3, max_shift=0.1, tol=0.001):
    """
    Find common sources given a set of sources and spike times

    Args:
        df1 (DataFrame): Data Frame containing spiking neuron activities (columns: 'source_id', 'spike_time')
        fsamp (float): Sampling frequecny (in Hz) of the binary spike trains
        t_start (float) : Start of the time window to be considered (in seconds)
        t_end (float): End of the time window to be considered (in seconds)
        theshold (float) : Common sources need to have a matching score higher than the theshold
        max_shift (float): Maximum delay between two sources (in seconds)
        tol (float): Common spikes need to be in the window [spike-tol, spike+tol]


    Returns:
        labels (ndarray): new labels of the sources
        match_matrix (ndarray): matching scores between all pairs of sources

    """
    
    units = sorted(df['source_id'].unique())
    n_source = len(units)
    labels = np.arange(n_source)
    match_matrix = np.identity(n_source)

    for i in np.arange(n_source):
        spikes_1 = df[df['source_id'] == units[i]]['spike_time'].values
        st1 = bin_spikes(spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end)
        
        for j in np.arange(i+1, n_source):
            spikes_2 = df[df['source_id'] == units[j]]['spike_time'].values
            st2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
            _ , shift = max_xcorr(st1, st2, max_shift=int(max_shift*fsamp))
            tp, _, _ = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol) 
            denom = max(len(spikes_1), len(spikes_2))
            match_score = tp / denom if denom > 0 else 0

            match_matrix[i,j] = match_score
            match_matrix[j,i] = match_score

            if match_score >= threshold:
                labels[j] = i

    return labels, match_matrix

def signal_based_quality_metrics(source, spikes, fsamp, min_peak_dist=0.01, match_dist=0.001):
    """
    Compute a set of signal based quality metrics

    Args:
        - source (ndarray): The predicted source
        - spikes (ndarray): Indices of the predicted spikes
        - fsamp : float Sampling frequency (Hz).
        - min_peak_distance_ms (float): Minimum distance between peaks (for peak detection), in seconds.
        - match_dist (float): Window size (± s) around predicted spikes to exclude from background.

    Returns
        - quality_metrics (dict): Dictonary of source quality metrics    
    
    """

    quality_metrics = {}

    # Number of Spikes
    quality_metrics['n_spikes'] = len(spikes)
    # Skewness
    quality_metrics['skew_val'] = skew(source)
    # Kurtosis
    quality_metrics['kurt_val'] = kurtosis(source)
    # Mean peak amplitude
    quality_metrics['peak_height'] = np.mean(source[spikes])
    # Coefficient of variation of the peaks
    quality_metrics['cov_peak'] = np.std(source[spikes])/np.mean(source[spikes])

    # Silhouette-like score
    sil, background_peaks = pseudo_sil_score(source, spikes, fsamp, 
                                             min_peak_dist=min_peak_dist, 
                                             match_dist=match_dist)
    
    quality_metrics['sil'] = sil

    pred_amps = source[spikes]
    back_amps = source[background_peaks]
    
    # Seperability metric
    quality_metrics['sep_prctile90'] = (np.percentile(pred_amps,10) - np.percentile(back_amps,90)) / np.mean(pred_amps)

    # Seperation in terms of standard deviations
    quality_metrics['sep_std'] = (np.mean(pred_amps) - np.mean(back_amps)) / np.std(pred_amps)

    # Pulse-to-noise ration (PNR)
    pnr, noise_indices = calc_pnr(source, spikes)
    quality_metrics['pnr'] = pnr

    # Z-score of the peak amplitude
    quality_metrics['z_score_height'] = np.mean(source[spikes]) / np.std(source[noise_indices])

    return quality_metrics

def pseudo_sil_score(source, spikes, fsamp, min_peak_dist=0.01, match_dist=0.001):
    """
    Computes a silhouette-like quality score for predicted spikes based on 
    peak detection and background spike amplitudes.

    Args:
        - source (np.ndarray): The source signal.
        - predicted_spikes (np.ndarray): Indices of predicted spike times.
        - fsamp : float Sampling frequency (Hz).
        - min_peak_distance_ms (float): Minimum distance between peaks (for peak detection), in seconds.
        - match_dist (float): Window size (± s) around predicted spikes to exclude from background.

    Returns:
        - sil (float): Silhouette-like quality score.
        - background_spikes (ndarray): Indices of the background spikes
        - centroids (tuple): Centrodis of the peak and noise cluster
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
    within = np.sum((pred_amps  - centroids[0])**2) if len(pred_amps) > 1 else 0.0
    between = np.sum((pred_amps  - centroids[1])**2)

    sil = (between - within) / max(between, within) if max(between, within) > 0 else 0.0

    return sil, background_spikes

def calc_pnr(source, spikes):
    """
    TODO Add description
    
    """

    # Calculate PNR
    signal_length = len(source)

    sig = source * abs(source)

    expanded_indices = set()
    for idx in spikes:
        for neighbor in [idx-1, idx, idx+1]:
            if 0 <= neighbor < signal_length:
                expanded_indices.add(neighbor)

    expanded_indices = np.array(sorted(expanded_indices))

    all_indices = np.arange(signal_length)
    noise_indices = np.setdiff1d(all_indices, expanded_indices)

    peak_cluster = sig[expanded_indices]**2
    noise_cluster = sig[noise_indices]**2

    pnr = 10 * np.log10(np.mean(peak_cluster) / np.mean(noise_cluster))

    return pnr, noise_indices

def get_basic_spike_statistics(spike_times, min_num_spikes=10):
    """
    Compute the mean firing rate (Hz) and the coefficient of variation
    of the interspike intervalls given a spike train

    Args:
        - spike_times (ndarray): Array of spike times (in seconds)
        - min_num_spikes (int): Minimum number of spikes required for the analysis

    Returns:
        - cov (float): Coefficient of variation of the interspike intervalls
        - mean_fr (float): Mean discharge rate of the neuron
    
    """

    cov = np.inf
    mean_fr = 0
    
    if len(spike_times) > min_num_spikes:
        # Get the interspike intervals
        isi  = np.diff(spike_times)
        # Reject periods where the neuron was inactive
        std_isi = np.std(isi)
        isi = isi[isi <= 10*std_isi]

        if len(isi) > 1:
            # Compute the CoV of the interspike intervals
            cov  = np.std(isi) / np.mean(isi)
            # Compute the mean discharge rate
            mean_fr = 1 / np.mean(isi)
    
    return cov, mean_fr




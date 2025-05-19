import numpy as np
import pandas as pd
from .evaluate import *
from ..algorithms.decomposition_routines import peel_off
from ..algorithms.pre_processing import bandpass_signals, notch_signals
from datetime import datetime

def signal_based_metrics(emg_data, sources, spikes_df, pipeline_sidecar, fsamp, datasetname, filename, target_muscle='n.a'):
    """
    Summarize signal and source based quality metrics as well as metadata in report cards

    Args:
        - emg_data (ndarray): EMG signal the decomposition was applied on (n_channels x time_samples)
        - sources (ndarray): Sources decomposed from the EMG data (n_sources x time_samples)
        - spikes_df (pd.DataFrame): Long formated dictonary of motor neuron spikes
        - pipeline_sidecar (dict): Log file of a BIDS decomposition derivative
        - fsamp (float): Sampling rate in Hz
        - datasetname (str): Name of the source dataset
        - filename (str): Name of the decomposed file
        - target_muscle (str): Muscle of interesst

    Returns:
        - global_report (pd.DataFrame): Summary of signal-based quality metrics and metadata
        - source_report (pd.DataFrame): Summary of source-specific perfomance metrics

    
    """

    # Algorithm name
    pipelinename = pipeline_sidecar['PipelineName']

    # Algorithm runtime
    runtime = get_runtime(pipeline_sidecar)

    # Time window the decomposition was run on
    t0, t1 = get_time_window(pipeline_sidecar, pipelinename)
    timeframe = [int(t0 * fsamp), int(t1 * fsamp)]
    
    # Compute the variance of the EMG signal that is explained by the decomposition
    explained_var, muap_rms = compute_reconstruction_error(sig=emg_data, 
                                                           spikes_df=spikes_df, 
                                                           fsamp=fsamp, 
                                                           timeframe=timeframe)
    
    # Summarize gloabl performance metrics together with metadata
    global_report = {'datasetname': [datasetname], 'filename': [filename],
                     'target_muscle': [target_muscle], 'runtime': [runtime], 
                      'explained_var': [explained_var]}
    
    global_report = pd.DataFrame(global_report)

    # Get a list of the extracted sources
    unique_labels = spikes_df['unit_id'].unique()

    if sources.shape[0] == 1:
        # Output an empty dataframe if no source was detected
        source_report = pd.DataFrame()
    else:
        source_report = []

        for i in np.arange(len(unique_labels)):
            # Extact spike times and indices
            spike_indices = spikes_df[spikes_df['unit_id'] == unique_labels[i]]['timestamp'].values.astype(int)
            spike_times = spikes_df[spikes_df['unit_id'] == unique_labels[i]]['spike_time'].values
            # Get spiking source statistics
            cov_isi, mean_dr = get_basic_spike_statistics(spike_times)
            # Compute a set of source-based quality metrics
            quality_metrics = signal_based_quality_metrics(sources[i,:], spike_indices, fsamp)
            source_report.append({
                'unit_id': int(unique_labels[i]),
                'datasetname': datasetname,
                'filename': filename,
                'target_muscle': target_muscle,
                'n_spikes': int(quality_metrics['n_spikes']),
                'sil': quality_metrics['sil'],
                'pnr': quality_metrics['pnr'],
                'peak_height': quality_metrics['peak_height'],
                'z_score': quality_metrics['z_score_height'],
                'cov_peak': quality_metrics['cov_peak'],
                'sep_prctile90': quality_metrics['sep_prctile90'],
                'sep_std': quality_metrics['sep_std'],
                'skew': quality_metrics['skew_val'],
                'kurt': quality_metrics['kurt_val'],
                'cov_isi': cov_isi,
                'mean_dr': mean_dr,
                'muap_rms': muap_rms[i]
                })
    source_report = pd.DataFrame(source_report)

    return global_report, source_report

def evaluate_spike_matches(df1, df2, t_start = 0, t_end = 60, tol=0.001, 
                           max_shift=0.1, fsamp = 2048, threshold=0.3, pre_matched=False):
    """
    Match spiking sources betwee two data sets.

    Args:
        df1 (DataFrame): Data Frame containing spiking neuron activities (columns: 'source_id', 'spike_time')
        df2 (DataFrame): Data Frame containing spiking neuron activities (columns: 'source_id', 'spike_time')
        t_start (float) : Start of the time window to be considered (in seconds)
        t_end (float): End of the time window to be considered (in seconds)
        tol (float): Common spikes need to be in the window [spike-tol, spike+tol]
        max_shift (float): Maximum delay between two sources (in seconds)
        fsamp (float): Sampling rate (in Hz) of the binary spike train
        theshold (float) : Common sources need to have a matching score higher than the theshold

    Returns:
        results (DataFrame): Table of matched units
        
    """
    source_labels_1 = sorted(df1['unit_id'].unique())
    source_labels_2 = sorted(df2['unit_id'].unique())
    used_labels = set()
    results = []

    for l1 in source_labels_1:
        spikes_1 = df1[df1['unit_id'] == l1]['spike_time'].values
        spikes_1 = spikes_1[(spikes_1 >= t_start) & (spikes_1 < t_end)]
        spike_train_1 = bin_spikes(spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end)
        best_match = None
        best_score = 0

        if pre_matched:
            l2 = l1
            spikes_2 = df2[df2['unit_id'] == l2]['spike_time'].values
            spikes_2 = spikes_2[(spikes_2 >= t_start) & (spikes_2 < t_end)]
            spike_train_2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
            _ , shift = max_xcorr(spike_train_1, spike_train_2, max_shift=int(max_shift*fsamp))
            #tp, fp, fn = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol)
            tp, fp, fn = match_spike_trains(spike_train_1,spike_train_2, shift=shift,tol=tol,fsamp=fsamp)
            best_score = 1 
            best_match = (l1, l2, tp, fp, fn, shift)

        else:
            for l2 in source_labels_2:
                if l2 in used_labels:
                    continue

                spikes_2 = df2[df2['unit_id'] == l2]['spike_time'].values
                spikes_2 = spikes_2[(spikes_2 >= t_start) & (spikes_2 < t_end)]
                spike_train_2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
                _ , shift = max_xcorr(spike_train_1, spike_train_2, max_shift=int(max_shift*fsamp))
                #tp, fp, fn = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol) 
                tp, fp, fn = match_spike_trains(spike_train_1,spike_train_2, shift=shift,tol=tol,fsamp=fsamp)
                denom = len(spikes_2)
                match_score = tp / denom if denom > 0 else 0

                if match_score > best_score:
                    best_score = match_score
                    best_match = (l1, l2, tp, fp, fn, shift)

        if best_match and best_score >= threshold:
            l1, l2, tp, fp, fn, shift = best_match
            results.append({
                'unit_id': l1,
                'unit_id_ref': l2,
                #'match_score': best_score,
                'delay_seconds': shift/fsamp,
                'TP': tp,
                'FN': fn,
                'FP': fp
            })
            used_labels.add(l2)
        else:
            # If no match was found, mark as unmatched
            results.append({
                'unit_id': l1,
                'unit_id_ref': None,
                #'match_score': 0,
                'delay_seconds': None,
                'TP': 0,
                'FN': 0,
                'FP': len(spikes_1)
            })   

    return pd.DataFrame(results)

def compute_reconstruction_error(sig, spikes_df, timeframe = None, win=0.05, fsamp=2048):
    """
    Compute the fraction of the variance of an EMG signal that 
    is explained by a decomposition output

    Args:
        - sig (ndarray): The emg data (channels x time_samples)
        - spikes_df (pd.DataFrame): Long formated dictonary of motor neuron spikes
        - timeframe (None or tuple): Time window the decomposition was applied to (start_idx, end_idx)
        - win (float): Plus/minus the duration in seconds of the window used to estimate the MUAP
        - fsamp (float): Sampling rate of the EMG signal in Hz   

    Returns:
        - explained_var (float): Fraction of variance explained by the decomposition
        - waveform_rms (ndarray): Relative RMS-amplitude of each source   
    
    """

    sig = bandpass_signals(sig, fsamp)
    sig = notch_signals(sig, fsamp)

    residual_sig = sig
    reconstructed_sig = np.zeros_like(sig)

    unique_labels = spikes_df['unit_id'].unique()

    df = spikes_df.copy()

    if timeframe is not None:
        sig = sig[:, timeframe[0]:timeframe[1]]
        residual_sig = residual_sig[:, timeframe[0]:timeframe[1]]
        reconstructed_sig = reconstructed_sig[:, timeframe[0]:timeframe[1]]
        df['timestamp'] = df['timestamp'] - timeframe[0]

    sig_rms = np.sqrt(np.mean(sig**2))
    waveform_rms = np.zeros(len(unique_labels))         

    for i in np.arange(len(unique_labels)):
        spike_indices = df[df['unit_id'] == unique_labels[i]]['timestamp'].values.astype(int)
        residual_sig, comp_sig, waveform = peel_off(residual_sig, spike_indices, win=win, fsamp=fsamp)
        reconstructed_sig += comp_sig
        waveform_rms[i] = np.sqrt(np.mean(waveform**2)) / sig_rms

    explained_var = 1 - np.var(residual_sig) / np.var(sig)    

    return explained_var, waveform_rms

def get_runtime(pipeline_sidecar):
    """
    Helper function to extract the runtime from
    an algorithm log file.

    Args:
        - pipeline_sidecar (dict): Log file of a BIDS decomposition derivative

    Returns:
        - runtime (float): Algorithm runtime in seconds    
    
    """

    t0 = pipeline_sidecar['Execution']['Timing']['Start']
    t1 = pipeline_sidecar['Execution']['Timing']['End']

    t0 = datetime.fromisoformat(t0)
    t1 = datetime.fromisoformat(t1)

    runtime = (t1 - t0).total_seconds()

    return runtime

def get_time_window(pipeline_sidecar, pipelinename):
    """
    Helper function to extract the time window the decomposition was
    peformed from a log file.

    Args:
        - pipeline_sidecar (dict): Log file of a BIDS decomposition derivative
        - pipelinename (str): Name of the algorithm generating the derivative

    Returns:
        - t0 (float): Start time in seconds 
        - t1 (float): End time in seconds     
    
    """

    if pipelinename == 'cbss':
        t0 = pipeline_sidecar['AlgorithmConfiguration']['start_time']
        t1 = pipeline_sidecar['AlgorithmConfiguration']['end_time']
    elif pipelinename == 'upperbound':
        t0 = pipeline_sidecar['AlgorithmConfiguration']['start_time']
        t1 = pipeline_sidecar['AlgorithmConfiguration']['end_time']    
    elif pipelinename == 'scd':
        t0 = pipeline_sidecar['AlgorithmConfiguration']['Config']['start_time']
        t1 = pipeline_sidecar['AlgorithmConfiguration']['Config']['end_time']
    else:
        raise ValueError('Invalid algorithm')   

    return t0, t1

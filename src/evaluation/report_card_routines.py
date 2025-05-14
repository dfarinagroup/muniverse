import numpy as np
import pandas as pd
from .evaluate import get_basic_spike_statistics, signal_based_quality_metrics
from ..algorithms.decomposition_routines import peel_off
from ..algorithms.pre_processing import bandpass_signals, notch_signals
from datetime import datetime

def summarize_signal_based_metrics(sources, spikes_df, muap_rms, fsamp, datasetname, filename, target_muscle='n.a'):
    """
    TODO Add description
    
    """

    unique_labels = spikes_df['unit_id'].unique()

    results = []

    for i in np.arange(len(unique_labels)):
        spike_indices = spikes_df[spikes_df['unit_id'] == unique_labels[i]]['timestamp'].values.astype(int)
        spike_times = spikes_df[spikes_df['unit_id'] == unique_labels[i]]['spike_time'].values
        cov_isi, mean_dr = get_basic_spike_statistics(spike_times)
        quality_metrics = signal_based_quality_metrics(sources[i,:], spike_indices, fsamp)
        results.append({
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
        
    return pd.DataFrame(results)

def compute_reconstruction_error(sig, spikes_df, timeframe = None, win=0.05, fsamp=2048):

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

    # if timeframe is not None:
    #     sig[:, :timeframe[0]] = 0
    #     sig[:, timeframe[1]:] = 0  
    #     residual_sig[:, :timeframe[0]] = 0
    #     residual_sig[:, timeframe[1]:] = 0    

    explained_var = 1 - np.var(residual_sig) / np.var(sig)    

    return explained_var, waveform_rms

def get_runtime(pipeline_sidecar):

    t0 = pipeline_sidecar['Execution']['Timing']['Start']
    t1 = pipeline_sidecar['Execution']['Timing']['End']

    t0 = datetime.fromisoformat(t0)
    t1 = datetime.fromisoformat(t1)

    runtime = (t1 - t0).total_seconds()

    return runtime

def get_global_metrics(explained_var, pipeline_sidecar, datasetname, filename, target_muscle='n.a'):

    runtime = get_runtime(pipeline_sidecar)

    results = {'datasetname': [datasetname], 'filename': [filename],
               'target_muscle': [target_muscle], 'runtime': [runtime], 
               'explained_var': [explained_var]}

    return pd.DataFrame(results)
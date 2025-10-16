"""
Generate report cards from decomposition outputs 
"""
import numpy as np
import argparse
from edfio import read_edf
import pandas as pd
import glob
import os
import json
import re

from muniverse.utils.data2bids import edf_to_numpy
from muniverse.evaluation.report_card_routines import *

DATA_DIR = "/rds/general/user/pm1222/ephemeral/muniverse/data/"
MUAPS_DIR = "/rds/general/user/pm1222/home/data/muapcache/"
INTERIM_DIR = "/rds/general/user/pm1222/ephemeral/muniverse/interim/"


def handle_neuromotion_spikes(spikes, fsamp):
    df = spikes.copy()
    df = df.rename(columns={'source_id': 'unit_id'})
    df['timestamp'] = df['spike_time']
    df['spike_time'] = df['spike_time'] / fsamp
    return df.sort_values('spike_time')


def handle_predicted_spikes(spikes, fsamp):
    df = spikes.copy()
    df = df.rename(columns={'source_id': 'unit_id'})
    df['timestamp'] = df['spike_time'] * fsamp
    df['timestamp'] = df['timestamp'].astype(int)
    return df.sort_values('spike_time')


def get_input_data(file_identifier, DSET):
    emg_file = glob.glob(os.path.join(DATA_DIR, f'{DSET}/**/{file_identifier}*_emg.edf'), recursive=True)[0]
    spikes_file = glob.glob(os.path.join(DATA_DIR, f'{DSET}/**/{file_identifier}*_spikes.tsv'), recursive=True)[0]
    simulation_sidecar = glob.glob(os.path.join(DATA_DIR, f'{DSET}/**/{file_identifier}*_simulation.json'), recursive=True)[0]
    emg_data = read_edf(emg_file)
    emg_data = edf_to_numpy(emg_data, np.arange(emg_data.num_signals))
    true_spikes_df = pd.read_csv(spikes_file, sep='\t')
    simulation_sidecar = json.load(open(simulation_sidecar))
    
    return emg_data, true_spikes_df, simulation_sidecar


def get_result_data(RESULT_DIR):
    sources = glob.glob(os.path.join(RESULT_DIR, '*_sources.npz'))[0]
    sources = np.load(sources, allow_pickle=True)['sources']

    try:
        spikes = glob.glob(os.path.join(RESULT_DIR, '*_timestamps.tsv'))[0]
        spikes = pd.read_csv(spikes, sep='\t')
    except:
        spikes = pd.DataFrame(columns=['source_id', 'spike_time'])

    pipeline_sidecar = glob.glob(os.path.join(RESULT_DIR, '*_metadata.json'))[0]    
    pipeline_sidecar = json.load(open(pipeline_sidecar))
    
    return sources, spikes, pipeline_sidecar


def parse_filename(filename):
    pattern = r'sub-(?P<subject_id>[^_]+)_ses-(?P<session_id>\d+)_task-(?P<muscle_name>[A-Z]{2,4})(?P<contraction_type>dynamic|isometric)(?P<movement_type>flexion|radial|extension|ulnar)(?P<contraction_profile>sinusoid|trapezoid|constant)(?P<mvc_percent>\d+)percentmvc(?:_run-(?P<run_id>\d+))?'
    match = re.match(pattern, filename)
    return match.groupdict()


def get_muaps(file_identifier):
    fileparts = parse_filename(file_identifier)
    
    if fileparts['movement_type'] == "flexion":
        mov = "Flexion-Extension"
    elif fileparts['movement_type'] == "radial":
        mov = "Radial-Ulnar-deviation"

    sub = fileparts['subject_id'][-1]
    muscle = fileparts['muscle_name']
    
    muap_file = f'subject_{sub}_{muscle}_{mov}_muaps.npy'
    muap_file = os.path.join(MUAPS_DIR, muap_file)
    print(f'Loading MUAPs from {muap_file}')
    muaps = np.load(muap_file)
    return muaps


def main():
    parser = argparse.ArgumentParser(description='Generate report card for a decomposition pipeline applied to a dataset')
    parser.add_argument('-d', '--dataset_name', help='Name of the dataset to process')
    parser.add_argument('-a', '--algorithm', choices=['scd', 'cbss', 'upperbound'], help='Algorithm to use for decomposition')
    parser.add_argument('-min_id', type=int, default=0, help='Minimum ID to process')
    parser.add_argument('-max_id', type=int, default=None, help='Maximum ID to process')

    # Parse function arguments
    args = parser.parse_args()
    datasetname = args.dataset_name
    algorithm = args.algorithm
    min_id = args.min_id
    max_id = args.max_id

    RESULTS_BASE = os.path.join(INTERIM_DIR, algorithm, datasetname)
    RESULTS_DIRS = glob.glob(os.path.join(RESULTS_BASE, '*/'))
    if max_id is None:
        max_id = len(RESULTS_DIRS) - 1

    global_report = pd.DataFrame()
    source_report = pd.DataFrame()

    for idx in range(min_id, max_id + 1):
        file_identifier = RESULTS_DIRS[idx].split('/')[-2]
        print(f'Processing {file_identifier}')
        
        # Load input data
        emg_data, true_spikes_df, simulation_sidecar = get_input_data(file_identifier, datasetname)
        fsamp = simulation_sidecar['InputData']['Configuration']['RecordingConfiguration']['SamplingFrequency']
        true_spikes_df = handle_neuromotion_spikes(true_spikes_df, fsamp)

        # Load results
        # Spikes are already time-shifted; need to shift sources
        predicted_sources, predicted_spikes_df, pipeline_sidecar = get_result_data(os.path.join(RESULTS_BASE, file_identifier))
        t_start, t_end = get_time_window(pipeline_sidecar, algorithm)
        if t_end < 0:
            dur = int(emg_data.shape[0] / fsamp)
            t_end += dur

        predicted_spikes_df = handle_predicted_spikes(predicted_spikes_df, fsamp)
        if algorithm == 'scd':
            predicted_sources = predicted_sources.T

        n_sources = predicted_spikes_df['unit_id'].nunique()
        if n_sources == 0:
            print(f'No sources found for {file_identifier}, skipping...')
            continue
        
        n_timestamps = emg_data.shape[0]
        shifted_sources = np.zeros((n_sources, n_timestamps))
        shifted_sources[:, int(t_start * fsamp):int(t_end * fsamp)] = predicted_sources
        pipeline_sidecar['PipelineName'] = algorithm

        my_global_report, my_source_report = signal_based_metrics(
            emg_data.T, shifted_sources, predicted_spikes_df, pipeline_sidecar, 
            fsamp=fsamp, datasetname=datasetname, filename=file_identifier,)

        # Compare the decomposition to reference spikes
        df = evaluate_spike_matches(predicted_spikes_df, true_spikes_df, t_start=t_start, t_end=t_end, fsamp=fsamp)
        my_source_report = pd.merge(my_source_report, df, on='unit_id')

        global_report = pd.concat([global_report, my_global_report], ignore_index=True)
        source_report = pd.concat([source_report, my_source_report], ignore_index=True)
        print(f'Finished analyzing {idx+1} out of {len(RESULTS_DIRS)} files')

    global_report.to_csv(os.path.join(RESULTS_BASE, f'report_card_globals_{min_id}_{max_id}.tsv'), sep='\t', index=False, header=True)
    source_report.to_csv(os.path.join(RESULTS_BASE, f'report_card_sources_{min_id}_{max_id}.tsv'), sep='\t', index=False, header=True)

if __name__ == '__main__':
    main()
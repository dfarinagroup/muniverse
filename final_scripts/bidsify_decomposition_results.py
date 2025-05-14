import argparse
import numpy as np
import pandas as pd
import json
import os
from edfio import *
from muniverse.data_preparation.data2bids import *
from pathlib import Path
import glob

def list_files(root, extension):
    files = list(root.rglob(f'*{extension}'))
    return files

def main():
    parser = argparse.ArgumentParser(description='Convert the output of a decomposition in BIDS format')
    parser.add_argument('-d', '--dataset_name', help='Name of the dataset to process')
    parser.add_argument('-a', '--algorithm', choices=['scd', 'cbss'], help='Algorithm to use for decomposition')
    parser.add_argument('-r', '--bids_root',  default='/rds/general/user/pm1222/ephemeral/muniverse/', help='Path to the muniverse datasets')
    parser.add_argument('-s', '--source_root', help='Path to the raw decomposition outputs')
    
    args = parser.parse_args()

    # Datset and Pipeline information
    datasetname = args.dataset_name
    pipelinename = args.algorithm
    root = args.bids_root
    source_root = args.source_root

    # Link to the source dataset
    source_dataset = bids_dataset(datasetname=datasetname, root=root + '/datasets/bids/')
    source_dataset.read()
    source_file_list = source_dataset.list_all_file('_emg.edf')

    # Get all decomposition log files
    files = list_files(Path(source_root),'log.json')

    skipped = 0
    for i in np.arange(len(files)):
        with open(str(files[i]), 'r') as f:
            pipeline_sidecar = json.load(f)
        
        # Find recording in dataset
        source_file_name = pipeline_sidecar['InputData']['FileName']
        print(files[i])
        matches = source_file_list[source_file_list['file_name'].str.endswith(source_file_name)]
        match_idx = matches.index.to_list()[0]

        # Link to BIDS recording 
        emg_recording = bids_emg_recording(data_obj=source_dataset)
        emg_recording.read_data_frame(source_file_list,match_idx)
        emg_recording.read()

        # fsamp = emg_recording.channels.loc[0, 'sampling_frequency']
        if pipelinename == 'scd':
            config = pipeline_sidecar['AlgorithmConfiguration']['Config']
        elif pipelinename == 'cbss':
            config = pipeline_sidecar['AlgorithmConfiguration']
        
        fsamp = config['sampling_frequency']
        n_samples = emg_recording.emg_data.signals[0].data.shape[0]

        # Extract time configuration
        start_time = config['start_time']
        end_time = config['end_time']
        start_idx = int(start_time * fsamp)
        end_idx = int(end_time * fsamp)

        # Initalize BIDS deivatives class
        my_derivative = bids_decomp_derivatives(pipelinename=pipelinename, 
                                                root=root + '/derivatives/bids/', 
                                                datasetname=datasetname, 
                                                subject=emg_recording.subject_id, 
                                                task=emg_recording.task.split('-')[1], 
                                                session=emg_recording.session, 
                                                run=emg_recording.run, 
                                                datatype=emg_recording.datatype)
        
        # Set pipeline sidecar file
        my_derivative.set_metadata('pipeline_sidecar', str(files[i]))
        
        # Get the predicted spikes
        # If algorithm is CBSS, the spikes are [source_id, spike_time (s)]
        # If algorithm is SCD, the spikes are [unit_id, timestamp (samples)]
        try:
            spikes_file = str(list_files(files[i].parent, '.tsv')[0])
            spikes_df = pd.read_csv(spikes_file, delimiter='\t')
            if pipelinename == 'cbss':
                spikes_df['spike_time'] = spikes_df['spike_time'] + start_time
                spikes_df['timestamp'] = (spikes_df['spike_time'] * fsamp).astype(int)
                spikes_df = spikes_df.rename(columns={'source_id': 'unit_id'})
            elif pipelinename == 'scd':
                spikes_df['timestamp'] += int(start_time*fsamp)
                spikes_df['spike_time'] = (spikes_df['timestamp']/fsamp).astype(int)
            
                # Ensure we have exactly the required columns in the correct order
                spikes_df = spikes_df[['unit_id', 'spike_time', 'timestamp']]
        except:
            print(f'No spikes file found for {source_file_name}, skipping...')

        # Get the predicted sources 
        # If algorithm is CBSS, the sources are "sources", (channels, samples)
        # If algorithm is SCD, the sources are "predicted_sources", (samples, channels)
        try:
            predicted_sources = str(list_files(files[i].parent, '.npz')[0])
            if pipelinename == 'scd':
                key = 'predicted_sources'
                predicted_sources = np.load(predicted_sources)
                predicted_sources = predicted_sources[key]
            elif pipelinename == 'cbss':
                key = 'sources'
                predicted_sources = np.load(predicted_sources)
                predicted_sources = predicted_sources[key].T

            shifted_sources = np.zeros((n_samples, predicted_sources.shape[1]))

            for j in np.arange(predicted_sources.shape[1]):
                shifted_sources[start_idx:end_idx,j] = predicted_sources[:,j] 
            
            print(shifted_sources.shape)
            my_derivative.set_data('source', shifted_sources, fsamp)
        except:
            print(f'No predicted sources file found for {source_file_name}, skipping...')
            skipped += 1
        
        # Write your results
        my_derivative.set_metadata('spikes', spikes_df)
        my_derivative.write() 

    print(f'Skipped {skipped} files')

if __name__ == '__main__':
    main()        


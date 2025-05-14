import argparse
import numpy as np
import pandas as pd
import json
import os
from edfio import *
from muniverse.evaluation.report_card_routines import *
from muniverse.data_preparation.data2bids import *
from muniverse.evaluation.evaluate import *
from pathlib import Path
import glob

def list_files(root, extension):
    files = list(root.rglob(f'*{extension}'))
    return files

def get_recording_info(source_file_name):
    splitname = source_file_name.split('_')

    if splitname[1].split('-')[0] == 'ses':
        sub = int(splitname[0].split('-')[1])
        ses = int(splitname[1].split('-')[1])
        task = splitname[2].split('-')[1]
        run = int(splitname[3].split('-')[1])
        data_type = splitname[4].split('.')[0]
    else:        
        sub = int(splitname[0].split('-')[1])
        task = splitname[1].split('-')[1]
        run = int(splitname[2].split('-')[1])
        ses = -1
        data_type = splitname[3].split('.')[0]

    return sub, ses, task, run, data_type

def get_time_window(pipeline_sidecar, pipelinename):

    if pipelinename == 'cbss':
        t0 = pipeline_sidecar['AlgorithmConfiguration']['start_time']
        t1 = pipeline_sidecar['AlgorithmConfiguration']['end_time']
    elif pipelinename == 'scd':
        t0 = pipeline_sidecar['AlgorithmConfiguration']['Config']['start_time']
        t1 = pipeline_sidecar['AlgorithmConfiguration']['Config']['end_time']
    else:
        raise ValueError('Invalid algorithm')   

    return t0, t1

def main():
    parser = argparse.ArgumentParser(description='Generate report card for a decomposition pipeline applied to a dataset')
    parser.add_argument('-d', '--dataset_name', help='Name of the dataset to process')
    parser.add_argument('-r', '--bids_root',  default='/Users/thomi/Documents/muniverse-data', help='Path to the muniverse datasets')
    parser.add_argument('-a', '--algorithm', choices=['scd', 'cbss'], help='Algorithm to use for decomposition')

    args = parser.parse_args()

    datasetname = args.dataset_name
    pipelinename = args.algorithm
    root = args.bids_root

    parent_folder = root + '/Benchmarks/' + datasetname + '-' + pipelinename

    # # Get all folders that are part of the benchmark
    # folders = [f for f in os.listdir(parent_folder)
    #        if os.path.isdir(os.path.join(parent_folder, f))]

    datatype = 'emg'

    global_report = pd.DataFrame()
    source_report = pd.DataFrame()


    files = list_files(Path(parent_folder), '_predictedsources.edf')
    filenames = [f.name for f in files]

    for j in np.arange(len(files)):

        # Extract the relevant information of one recording
        sub, ses, task, run, _ = get_recording_info(filenames[j])

        # Get the raw data
        my_emg_data = bids_emg_recording(root=root + '/Datasets/', 
                                            datasetname=datasetname, 
                                            subject=sub, 
                                            task=task, 
                                            session=ses, 
                                            run=run, 
                                            datatype=datatype)
        
        my_emg_data.read()

        channel_idx = np.asarray(my_emg_data.channels[my_emg_data.channels['type'] == 'EMG'].index).astype(int)

        emg_data = edf_to_numpy(my_emg_data.emg_data, channel_idx)

        # Extract some metadata
        fsamp = float(my_emg_data.channels.loc[0, 'sampling_frequency'])
        target_muscle = my_emg_data.channels.loc[0, 'target_muscle']
        if datasetname == 'Caillet_et_al_2023':
            target_muscle = 'Tibialis Anterior'            

        # Get the decomposition
        my_derivative = bids_decomp_derivatives(pipelinename=pipelinename, 
                                                root=root + '/Benchmarks/', 
                                                datasetname=datasetname, 
                                                subject=sub, 
                                                task=task, 
                                                session=ses, 
                                                run=run, 
                                                datatype=datatype)
        
        my_derivative.read()

        t0, t1 = get_time_window(my_derivative.pipeline_sidecar, pipelinename)

        # Get global report
        my_global_report = get_global_metrics(emg_data=emg_data.T, 
                                              spikes_df=my_derivative.spikes, 
                                              fsamp=fsamp, 
                                              pipeline_sidecar=my_derivative.pipeline_sidecar,
                                              t_win = [t0, t1],
                                              datasetname=datasetname, 
                                              filename=filenames[j], 
                                              target_muscle=target_muscle
                                              )


        # Summarize all sources
        sources = edf_to_numpy(my_derivative.source,np.arange(my_derivative.source.num_signals))
        my_source_report = summarize_signal_based_metrics(sources=sources.T, 
                                                          spikes_df=my_derivative.spikes, 
                                                          fsamp=fsamp,
                                                          datasetname=datasetname,
                                                          filename=filenames[j],
                                                          target_muscle=target_muscle
                                                          )


        # If availible get get ground truth / reference decomposition
        if datasetname == 'Grison_et_al_2025':
            my_ref_derivative = bids_decomp_derivatives(pipelinename='reference', 
                                                        root=root + '/Benchmarks/', 
                                                        datasetname=datasetname, 
                                                        subject=sub, 
                                                        task=task, 
                                                        session=ses, 
                                                        run=run, 
                                                        datatype=datatype)
            
            my_ref_derivative.read()
            
            #start_time = my_derivative.pipeline_sidecar['AlgorithmConfiguration']['Config']['start_time']
            #end_time = my_derivative.pipeline_sidecar['AlgorithmConfiguration']['Config']['end_time']
            
            df = evaluate_spike_matches(my_derivative.spikes, my_ref_derivative.spikes, 
                                        t_start = t0, 
                                        t_end = t1)
            
            my_source_report = pd.merge(my_source_report, df, on='unit_id')

        global_report = pd.concat([global_report, my_global_report], ignore_index=True)
        source_report = pd.concat([source_report, my_source_report], ignore_index=True)
        print(f'Finished analyzing {j+1} out of {len(files)} files')
        
    global_report.to_csv(parent_folder + '/report_card_globals.tsv', sep='\t', index=False, header=True)
    source_report.to_csv(parent_folder + '/report_card_sources.tsv', sep='\t', index=False, header=True)

if __name__ == '__main__':
    main()        


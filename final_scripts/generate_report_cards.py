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
    """
    Helper function to list all files in folder "root" with an extension "extension"
    
    """
    files = list(root.rglob(f'*{extension}'))
    return files

def get_recording_info(source_file_name):
    """
    Helper function to extract metadata from a BIDS recording filename

    """
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

def handle_neuromotion_spikes(spikes, fsamp):
    """
    Handle the spikes from neuromotion data
    Currently they are stored as [source_id, spike_time]
    Derivatives need [unit_id, spike_time, timestamp]
    """
    # If input is already a DataFrame, just rename columns and add timestamp
    df = spikes.copy()
    df = df.rename(columns={'source_id': 'unit_id'})
    df['timestamp'] = df['spike_time']
    df['spike_time'] = df['spike_time'] / fsamp
    return df.sort_values('spike_time')

def main():
    """
    Main script controlable via the CLI
    
    """
    parser = argparse.ArgumentParser(description='Generate report card for a decomposition pipeline applied to a dataset')
    parser.add_argument('-d', '--dataset_name', help='Name of the dataset to process')
    parser.add_argument('-r', '--bids_root',  default='/rds/general/user/pm1222/ephemeral/muniverse/', help='Path to the muniverse datasets')
    parser.add_argument('-a', '--algorithm', choices=['scd', 'cbss', 'upperbound'], help='Algorithm to use for decomposition')
    parser.add_argument('-g', '--ground_truth', default='none', choices=['none', 'expert_ref', 'simulation'], help='Type of ground-truth reference')

    # Parse function arguments
    args = parser.parse_args()
    datasetname = args.dataset_name
    pipelinename = args.algorithm
    root = args.bids_root
    ground_truth = args.ground_truth

    # Path to the BIDS derivative dataset
    parent_folder = root + '/derivatives/bids/temp' + datasetname + '-' + pipelinename
    # Identify all derivatives that are part of the dataset
    files = list_files(Path(parent_folder), '_predictedsources.edf')
    filenames = [f.name for f in files]

    # Initalize the report cards
    global_report = pd.DataFrame()
    source_report = pd.DataFrame()
    
    # Set the datatype to EMG
    datatype = 'emg'

    for j in np.arange(len(files)):

        # Extract the relevant information of one recording
        sub, ses, task, run, _ = get_recording_info(filenames[j])

        # Initialize BIDS-EMG recording
        if ground_truth == 'simulation':
            my_emg_data = bids_neuromotion_recording(
                root=root + 'datasets/bids/', # TODO: handle this using Paths instead
                datasetname=datasetname,
                subject=sub,
                task=task,
                session=ses,
                run=run,
                datatype=datatype)
            
            # Extract the EMG data for neuromotion data # TODO: mimic lines 123-131: handle channels.tsv appropriately for neuromotion datasets
            my_emg_data.read()
            emg_data = edf_to_numpy(my_emg_data.emg_data, np.arange(my_emg_data.emg_data.num_signals)) 
            fsamp = my_emg_data.simulation_sidecar['InputData']['Configuration']['RecordingConfiguration']['SamplingFrequency']
            target_muscle = my_emg_data.simulation_sidecar['InputData']['Configuration']['MovementConfiguration']['TargetMuscle']
            rest_dur = my_emg_data.simulation_sidecar['InputData']['Configuration']['MovementConfiguration']['MovementProfileParameters']['RestDuration']
            dur = my_emg_data.simulation_sidecar['InputData']['Configuration']['MovementConfiguration']['MovementProfileParameters']['MovementDuration']
            print(f'Extracting recording {filenames[j]}, duration {dur - 2*rest_dur} seconds')
        
        else:
            my_emg_data = bids_emg_recording(
                root=root + 'datasets/bids/', 
                datasetname=datasetname, 
                subject=sub, 
                task=task, 
                session=ses, 
                run=run, 
                datatype=datatype)
            
            # Extract the EMG data
            my_emg_data.read()
            channel_idx = np.asarray(my_emg_data.channels[my_emg_data.channels['type'] == 'EMG'].index).astype(int)
            emg_data = edf_to_numpy(my_emg_data.emg_data, channel_idx)

            # Extract some relevant metadata from the BIDS-EMG dataset
            fsamp = float(my_emg_data.channels.loc[0, 'sampling_frequency'])
            target_muscle = my_emg_data.channels.loc[0, 'target_muscle'] 
            if datasetname == 'Caillet_et_al_2023':
                target_muscle = 'Tibialis Anterior'

        # BIDS decomposition derivative
        my_derivative = bids_decomp_derivatives(pipelinename=pipelinename, 
                                                root=root + '/derivatives/bids/temp', 
                                                datasetname=datasetname, 
                                                subject=sub, 
                                                task=task, 
                                                session=ses, 
                                                run=run, 
                                                datatype=datatype)
        
        my_derivative.read()

        # Summarize all sources
        sources = edf_to_numpy(my_derivative.source,np.arange(my_derivative.source.num_signals))

        # Compute global and source based metrics (always possible)
        my_global_report, my_source_report = signal_based_metrics(emg_data=emg_data.T, 
                                                                  sources=sources.T, 
                                                                  spikes_df=my_derivative.spikes, 
                                                                  pipeline_sidecar=my_derivative.pipeline_sidecar, 
                                                                  fsamp=fsamp, 
                                                                  datasetname=datasetname, 
                                                                  filename=filenames[j], 
                                                                  target_muscle=target_muscle)


        # If availible get get ground truth / reference decomposition
        if ground_truth == 'expert_ref':
            # BIDS derivative of the reference decomposition
            my_ref_derivative = bids_decomp_derivatives(pipelinename='reference', 
                                                        root=root + '/derivatives/bids/temp', 
                                                        datasetname=datasetname, 
                                                        subject=sub, 
                                                        task=task, 
                                                        session=ses, 
                                                        run=run, 
                                                        datatype=datatype)
            
            my_ref_derivative.read()

            # Time frame the decomposition was applied to
            t0, t1 = get_time_window(my_derivative.pipeline_sidecar, pipelinename)

            # Comapre the decomposition to reference spikes            
            df = evaluate_spike_matches(my_derivative.spikes, my_ref_derivative.spikes, 
                                        t_start = t0, t_end = t1, fsamp=fsamp)
            # Add the output to the report card
            my_source_report = pd.merge(my_source_report, df, on='unit_id')

        elif ground_truth == 'simulation':
            gt_spikes = handle_neuromotion_spikes(my_emg_data.spikes, fsamp) # TODO: this should be incorporated into the bids_neuromotion_recording class

            # Time frame the decomposition was applied to
            t0, t1 = get_time_window(my_derivative.pipeline_sidecar, pipelinename)
            t1 += dur

            # Compare the decomposition to reference spikes 
            if pipelinename == 'upperbound':
                # If we save all sources, in upperbound we can simplify source matching as it is already known
                df = evaluate_spike_matches(my_derivative.spikes, gt_spikes, 
                                            t_start = t0, t_end = t1, fsamp=fsamp,
                                            pre_matched=True)
            else:
                df = evaluate_spike_matches(my_derivative.spikes, gt_spikes, 
                                            t_start = t0, t_end = t1, fsamp=fsamp)
            
            # Add the output to the report card
            my_source_report = pd.merge(my_source_report, df, on='unit_id')

        # Update report cards
        global_report = pd.concat([global_report, my_global_report], ignore_index=True)
        source_report = pd.concat([source_report, my_source_report], ignore_index=True)
        print(f'Finished analyzing {j+1} out of {len(files)} files')

    # Save the results    
    global_report.to_csv(parent_folder + '/report_card_globals.tsv', sep='\t', index=False, header=True)
    source_report.to_csv(parent_folder + '/report_card_sources.tsv', sep='\t', index=False, header=True)

if __name__ == '__main__':
    main()        


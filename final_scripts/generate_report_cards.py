import argparse
import numpy as np
import pandas as pd
import json
import os
from edfio import *
from muniverse.evaluation.report_card_routines import *
from muniverse.data_preparation.data2bids import *
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

def main():
    parser = argparse.ArgumentParser(description='Convert the output of a decomposition in BIDS format')
    parser.add_argument('-r', '--bids_root',  default='/Users/thomi/Documents/muniverse-data', help='Path to the muniverse datasets')
    
    args = parser.parse_args()

    root = args.bids_root

    parent_folder = root + '/Benchmarks/'

    folders = [f for f in os.listdir(parent_folder)
           if os.path.isdir(os.path.join(parent_folder, f))]

    datatype = 'emg'

    for i in np.arange(len(folders)):
        splitname = folders[i].split('-')
        datasetname = splitname[0]
        pipelinename = splitname[1]

        if datasetname == 'Grison_et_al_2025':
            fsamp = 10240
        else:
            fsamp = 2048


        files = list_files(Path(parent_folder + folders[i]), '_predictedsources.edf')
        filenames = [f.name for f in files]

        for j in np.arange(len(files)):
            sub, ses, task, run, _ = get_recording_info(filenames[j])

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

            my_derivative = bids_decomp_derivatives(pipelinename=pipelinename, 
                                                    root=root + '/Benchmarks/', 
                                                    datasetname=datasetname, 
                                                    subject=sub, 
                                                    task=task, 
                                                    session=ses, 
                                                    run=run, 
                                                    datatype=datatype)
            
            my_derivative.read()

            global_report = get_global_metrics(emg_data.T, my_derivative.spikes, fsamp, my_derivative.pipeline_sidecar)
            outputname = (my_derivative.root + my_derivative.datapath +
                    filenames[j].split('_predictedsources.edf')[0] + '_global_report.tsv')
            global_report.to_csv(outputname, sep='\t', index=False, header=True)

            sources = edf_to_numpy(my_derivative.source,np.arange(my_derivative.source.num_signals))
            source_report = summarize_signal_based_metrics(sources.T, my_derivative.spikes, fsamp=fsamp)

            outputname = (my_derivative.root + my_derivative.datapath +
                    filenames[j].split('_predictedsources.edf')[0] + '_source_report.tsv')
            source_report.to_csv(outputname, sep='\t', index=False, header=True)
            print('bla')


if __name__ == '__main__':
    main()        


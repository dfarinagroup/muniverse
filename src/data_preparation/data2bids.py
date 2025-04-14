import numpy as np
import os
import tarfile as tf
import xml.etree.ElementTree as ET
import json
import pandas as pd
from edfio import *
#import mne
#from mne_bids import BIDSPath, write_raw_bids

def open_otb(inputname,ngrid):
    # Extract data and metadata from OTB+ File

    # 
    filename = inputname.split('/')[-1]
    temp_dir = os.path.join('./', 'temp_tarholder')
    # make a temporary directory to store the data of the otb file if it doesn't exist yet
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    # Open the .tar file and extract all data
    with tf.open(inputname, 'r') as emg_tar:
        emg_tar.extractall(temp_dir)

    # Extract file names from .tar directory
    sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
    trial_label_sig = sig_files[0]  # only one .sig so can be used to get the trial name (0 index list->string)
    trial_label_xml = trial_label_sig.split('.')[0] + '.xml'
    trial_label_sig = os.path.join(temp_dir, trial_label_sig)
    trial_label_xml = os.path.join(temp_dir, trial_label_xml)
    sip_files = [f for f in os.listdir(temp_dir) if f.endswith('.sip')]

    # read the metadata xml file 
    with open(trial_label_xml, encoding='utf-8') as file:
        xml=ET.fromstring(file.read())

    # Get the device info
    device_info = xml.attrib

    # Get the adapter info 
    adapter_info = xml.findall('.//Adapter')

    nADbit = int(device_info['ad_bits'])
    nchans  = int(device_info['DeviceTotalChannels'])
    # read in the EMG trial data
    emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
    emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
    emg_data = emg_data.astype(float)

    # initalize data vector
    data = np.zeros((emg_data.shape[1], ngrid*64+len(sip_files)))

    # initalize vector of recorded units
    ch_units = []

    # convert the data from bits to microvolts
    for i in range(ngrid*64):
        data[:,i] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit)))
        ch_units.append('uV')

    # Get data and metadata from the aux input channels
    aux_info = dict()

    for i in range(len(sip_files)):
        # Get metadata
        tmp = sip_files[i]
        tmp = tmp.split('.')[0] + '.pro'
        tmp = os.path.join(temp_dir, tmp)
        with open(tmp, encoding='utf-8') as file:
            xml=ET.fromstring(file.read())

        aux_info[i] = {child.tag: child.text for child in xml}
        ch_units.append(aux_info[i]['unity_of_measurement'])
        
        # get data
        trial_label_sip = os.path.join(temp_dir, sip_files[i])
        aux_data = np.fromfile(open(trial_label_sip),dtype='float64')
        aux_data = aux_data[0:data.shape[0]]
        data[:,i+ngrid*64] = aux_data

    # Get the subject info
    with open(os.path.join(temp_dir, 'patient.xml'), encoding='utf-8') as file:
        xml=ET.fromstring(file.read())  

    subject_info = {child.tag: child.text for child in xml}      

    # Remove .tar folder
    for filename in os.listdir(temp_dir):
        file = os.path.join(temp_dir, filename)
        if os.path.isfile(file):
            os.remove(file)

    os.rmdir(temp_dir)

    metadata = {
                'device_info': device_info, 'adapter_info': adapter_info,
                'aux_info': aux_info, 'subject_info': subject_info, 'units': ch_units
                }

    return (data, metadata)

def format_otb_channel_metadata(data,metadata,ngrids):
    ch_names = ['Ch'+str(i) for i in np.arange(1,data.shape[1]+1)]
    units = metadata['units']
    ch_type = []
    low_cutoff = []
    high_cutoff = []
    sampling_frequency = []
    signal_electrode = []
    grid_name = []
    group = []
    reference = []
    target_muscle = []
    interelectrode_distance = []
    description = []

    # Get channel metadata
    for i in np.arange(ngrids):    
        channel_metadata = metadata['adapter_info'][i].findall('.//Channel')
        for j in np.arange(64):
            ch_type.append('EMG')
            low_cutoff.append(int(metadata['adapter_info'][i].attrib['LowPassFilter']))
            high_cutoff.append(int(metadata['adapter_info'][i].attrib['HighPassFilter']))
            sampling_frequency.append(int(metadata['device_info']['SampleFrequency']))
            signal_electrode.append('E' + str(j+1))
            grid_name.append(channel_metadata[j].attrib['ID'])
            group.append('Grid'+ str(i+1))
            reference.append('R1')
            target_muscle.append(channel_metadata[j].attrib['Muscle'])
            tmp = channel_metadata[j].attrib['Description']
            tmp = tmp.split('Array ')[-1]
            tmp = tmp.split((' i.e.d.'))[0]
            interelectrode_distance.append(tmp)
            description.append('Monopolar EMG')

    for i in np.arange(len(metadata['aux_info'])):
        ch_type.append('MISC')
        low_cutoff.append('n/a')
        high_cutoff.append('n/a')
        sampling_frequency.append(int(metadata['aux_info'][i]['fsample']))
        signal_electrode.append('n/a')
        grid_name.append('n/a')
        group.append('n/a')
        reference.append('n/a')
        target_muscle.append('n/a')
        interelectrode_distance.append('n/a')
        description.append(metadata['aux_info'][i]['description'])

    ch_metadata = {
        'name': ch_names, 'type': ch_type, 'unit': units,
        'description': description, 'sampling_frequency': sampling_frequency,
        'signal_electrode': signal_electrode, 'reference_electrode': reference,
        'group': group, 'target_muscle': target_muscle, 'interelectrode_distance': interelectrode_distance,
        'grid_name': grid_name, 'low_cutoff': low_cutoff, 'high_cutoff': high_cutoff
    }

    return(ch_metadata)    



def make_channel_tsv(bids_path, channel_metadata):
    # Make *_channels.tsv file
    # 
    # channel_metadata (dict) needs to contain the  
    # followinh essential keys in correct order:
    #   [0] name (string) 
    #   [1] type (string)
    #   [2] units (string) 

    if not isinstance(channel_metadata, dict):
        raise ValueError('channel_metadata is not class <dict>')

    keys = list(channel_metadata.keys())[0:3]    
    if not keys == ['name', 'type', 'unit']:
        raise ValueError('essential keys are missing or incorrectly ordered')
   
    path = bids_path['root'] + '/' +  bids_path['subject'] + '/' + bids_path['datatype'] + '/' 
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + 'channels'

    df = pd.DataFrame(data=channel_metadata)
    df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)

    return()

def make_electrode_tsv(bids_path, el_metadata):
    # Make *_electrodes.tsv file
    # 
    # Essentials (must be correctly ordered): 
    #   - name (string) 
    #   - x (number)
    #   - y (number)
    #   - z (number) -- if exists
    #   - coordinate_system (string) 

    # ToDo: Add check routines

    path = bids_path['root'] + '/' +  bids_path['subject'] + '/' + bids_path['datatype'] + '/' 
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + 'electrodes'

    df = pd.DataFrame(data=el_metadata)
    df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)

    return()

def make_emg_json(bids_path, emg_metadata):
    # Make *_emg.json file
    # 
    # Essentials: 
    #   - EMGPlacemnetScheme (string) 
    #   - EMGReference (string)
    #   - SamplingFrequency (number) 
    #   - PowerLineFrequency (number or "n/a"), 
    #   - SoftwareFilters (object of objects or "n/a")
    #   - TaskName (string)

    # ToDo: Test if essential keys are defined

    path = bids_path['root'] + '/' +  bids_path['subject'] + '/' + bids_path['datatype'] + '/' 
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + bids_path['datatype']

    with open(path + name + '.json', 'w') as f:
        json.dump(emg_metadata, f)

    return()

def make_coordinate_system_json(bids_path, coordsystem_metadata):
    # Make *_coordsystem.json file
    # 
    # Essentials: 
    #   - EMGCoordinateSystem (string) 
    #   - EMGCoordinateUnits (string)

    path = bids_path['root'] + '/' +  bids_path['subject'] + '/' + bids_path['datatype'] + '/' 
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + 'coordsystem'

    with open(path + name + '.json', 'w') as f:
        json.dump(coordsystem_metadata, f)

    return()

def make_participant_tsv(bids_path, subject_metadata):
    # Make participants.tsv file
    # 
    # subject_metadata is a dictonary containing the essential fields: 
    #   - name (string) 
    #   - age (number or "n/a")
    #   - sex (string or "n/a")
    #   - hand (string or "n/a") 
    #   - weight (string or "n/a") 
    #   - height (string or "n/a")

    # ToDo: Check if the essential fields are defined

    filename = bids_path['root'] + '/' + 'participants.tsv'
     
    if os.path.isfile(filename):
        df1 = pd.read_table(filename)
        df2 = pd.DataFrame(data=subject_metadata, index=[0])
        frames = [df1, df2]
        df = pd.concat(frames)
        df = df.drop_duplicates(subset='name', keep='first')
        df.to_csv(filename, sep='\t', index=False, header=True)
    else:
        df = pd.DataFrame(data=subject_metadata, index=[0])
        df.to_csv(filename, sep='\t', index=False, header=True)

    return()

def make_participant_json(bids_path,data_type):
    # Make participants.json file

    if data_type == 'simulation':
        metadata = {'name': {'Description': 'Unique subject identifier'},
                    'generated by': {'Description': 'This data set contains simulated data',
                                    'string': 'Software used to generate the data'}  
                    }
    else:    
        metadata = {'name': {'Description': 'Unique subject identifier'},
                    'age': {'Description': 'Age of the participant at time of testing', 
                            'Unit': 'years'},
                    'sex': {'Description': 'Biological sex of the participant',
                            'Levels': {'F': 'female', 'M': 'male'}},
                    'handedness': {'Description': 'handedness of the participant as reported by the participant',
                            'Levels': {'L': 'left', 'R': 'right'}},        
                    'weight': {'Description': 'Body weight of the participant', 
                            'Unit': 'kg'},
                    'height': {'Description': 'Body height of the participant', 
                            'Unit': 'm'}                
                    }
        
        filename = bids_path['root'] + '/' + 'participants.json'

        with open(filename, 'w') as f:
            json.dump(metadata, f)
    
    return()

def make_dataset_description_json(bids_path, metadata):
    # Make dataset_description.json 
    
    # Essentials:
    #   - Name (string)
    #   - BIDS Version (string)

    filename = bids_path['root'] + '/' + 'dataset_description.json'

    with open(filename, 'w') as f:
        json.dump(metadata, f)

    return()

def make_dataset_readme():
    # ToDo

    return()

# Todo: Include CITATION.cff?

def write_edf(data, fsamp, ch_names, bids_path):
    # basic version, one could add more metadata, e.g., see https://edfio.readthedocs.io/en/stable/examples.html

    # Get duration of the signal in seconds
    seconds = np.ceil(data.shape[0]/fsamp)
    # Add zeros to the signal such that the total length is in full seconds
    signal = np.zeros([int(seconds*fsamp), data.shape[1]])
    signal[0:data.shape[0],:] = data

    edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp, label=ch_names[0])])

    for i in np.arange(1,signal.shape[1]):
        new_signal = EdfSignal(signal[:,i], sampling_frequency=fsamp, label=ch_names[i])
        edf.append_signals(new_signal)

    path = bids_path['root'] + '/' +  bids_path['subject'] + '/' + bids_path['datatype'] + '/'  
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + bids_path['datatype']

    edf.write(path + name + '.edf')
    return()

def make_bids_path(subject, task, datatype, root):
    # 
    # ToDo: Make some form of validation?
    #
    bids_path_info = {'subject': 'sub' + '-' + str(subject).zfill(2),
                      'task': 'task-' + task,
                      'datatype': datatype,
                      'root': root}
    
    # make new folder
    newpath = root + '/' + bids_path_info['subject'] + '/' + datatype
    if not os.path.exists(newpath):
        os.makedirs(newpath)  

    return(bids_path_info)



import numpy as np
import os
import tarfile as tf
import xml.etree.ElementTree as ET
import json
import pandas as pd
from edfio import *

def open_otb(inputname,ngrid):
    """
    Reads otb+ files and outputs stored data and metadata

    Args:
        inputname (str): name and path of the inputfile, e.g. '/this/is/mypath/filename.otb+'
        ngrid (int): number of emg arrays used in the measurement

    Returns:
        data (ndarray): array of recorded data (samples x channels)
        metadata (dict): metadata of the recording
    """

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
    """
    Extract channel metadata given the output of the open_otb function

    Args:
        data (ndarray): array of recorded data (samples x channels)
        metadata (dict): metadata of the recording

    Returns:
        ch_metadata (dict): metadata associated with the individual channels
    """

    # Initalize lists for each metadata field 
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

    # Loop over all EMG channels
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

    # Loop over non-EMG channels
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

    # Output the channel metadata as dictonary
    ch_metadata = {
        'name': ch_names, 'type': ch_type, 'unit': units,
        'description': description, 'sampling_frequency': sampling_frequency,
        'signal_electrode': signal_electrode, 'reference_electrode': reference,
        'group': group, 'target_muscle': target_muscle, 'interelectrode_distance': interelectrode_distance,
        'grid_name': grid_name, 'low_cutoff': low_cutoff, 'high_cutoff': high_cutoff
    }

    return(ch_metadata)    

def make_channel_tsv(bids_path, channel_metadata):
    """
    Generate a BIDS compatible *_channels.tsv file

    Args:
        bids_path (dict): filename and filepath information
        channel_metadata (dict): Channel metadata with essential keys
            - name (string)
            - type (string)
            - units (string)

    """

    # Check if the essential keys exist (Note: ordering matters)
    keys = list(channel_metadata.keys())[0:3]    
    if not keys == ['name', 'type', 'unit']:
        raise ValueError('essential keys are missing or incorrectly ordered')
   
    # Get a BIDS compatible path and filename
    path = bids_path['datapath'] 
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + 'channels'

    # Convert metadata into a pandas data frame and save tsv-file
    df = pd.DataFrame(data=channel_metadata)
    df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)

    return()

def make_electrode_tsv(bids_path, el_metadata):
    """
    Generate a a BIDS compatible *_electrodes.tsv file

    Args:
        bids_path (dict): filename and filepath information
        el_metadata (dict): electrode metadata with essential keys
            - name (string)
            - x (float)
            - y (float)
            - z (float)
            - coordinate_system (string)

    """

    # Check if the essential keys exist (ordering matters)
    if 'z' in el_metadata:
        keys = list(el_metadata.keys())[0:5]    
        if not keys == ['name', 'x', 'y', 'z', 'coordinate_system']:
            raise ValueError('essential keys are missing or incorrectly ordered')
    else:    
        keys = list(el_metadata.keys())[0:4]    
        if not keys == ['name', 'x', 'y', 'coordinate_system']:
            raise ValueError('essential keys are missing or incorrectly ordered')

    # Get a BIDS compatible path and filename
    path = bids_path['datapath'] 
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + 'electrodes'

    # Convert metadata into a pandas data frame and save tsv-file
    df = pd.DataFrame(data=el_metadata)
    df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)

    return()

def make_emg_json(bids_path, emg_metadata):
    """
    Generate a a BIDS compatible *_emg.json file

    Args:
        bids_path (dict): filename and filepath information
        emg_metadata (dict): metadata with essential keys
            - EMGPlacementScheme (str)
            - EMGReference (str)
            - SamplingFrequency (float)
            - PowerLineFrequency (float or "n/a")
            - SoftwareFilters (dict or "n/a")
            - TaskName (str)

    """

    # Check if the essential keys exist
    essentials = ['EMGPlacementScheme', 'EMGReference', 'SamplingFrequency',
                  'PowerLineFrequency', 'SoftwareFilters', 'TaskName']
    
    for i in np.arange(len(essentials)):
        if essentials[i] not in emg_metadata:
            raise ValueError('essential keys are missing')

    # Get a BIDS compatible path and filename
    path = bids_path['datapath']
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + bids_path['datatype']

    # Store the metadata in a json file
    with open(path + name + '.json', 'w') as f:
        json.dump(emg_metadata, f)

    return()

def make_coordinate_system_json(bids_path, coordsystem_metadata):
    """
    Generate a a BIDS compatible *_coordsystem.json file

    Args:
        bids_path (dict): filename and filepath information
        coordsystem_metadata (dict): metadata with essential keys
            - EMGCoordinateSystem (str)
            - EMGCoordinateUnits (str)

    """

    # Check if the essential keys exist
    essentials = ['EMGCoordinateSystem', 'EMGCoordinateUnits']
    
    for i in np.arange(len(essentials)):
        if essentials[i] not in coordsystem_metadata:
            raise ValueError('essential keys are missing')

    # Get a BIDS compatible path and filename
    path = bids_path['datapath']
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + 'coordsystem'

    # Store the metadata in a json file
    with open(path + name + '.json', 'w') as f:
        json.dump(coordsystem_metadata, f)

    return()

def make_participant_tsv(bids_path, subject_metadata):
    """
    Generate a a BIDS compatible participants.tsv file

    Args:
        bids_path (dict): filename and filepath information
        subject_metadata (dict): metadata with essential keys
            - name (str) 
            - age (float or "n/a")
            - sex (str)
            - hand (str)
            - weight (float or "n/a")
            - height (float or "n/a") 
    """

    # Check if the essential keys exist
    essentials = ['name', 'age', 'sex', 'hand', 'weight', 'height']
    
    for i in np.arange(len(essentials)):
        if essentials[i] not in subject_metadata:
            raise ValueError('essential keys are missing')

    # Get a BIDS compatible path and filename
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
    """
    Generate a a BIDS compatible participants.json file

    Args:
        bids_path (dict): filename and filepath information
        data_type (str): 'simulation' or 'experimental'

    """

    # Hardcoded dictonary 
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
                            'Levels': {'F': 'female', 'M': 'male', 'O': 'other'}},
                    'handedness': {'Description': 'handedness of the participant as reported by the participant',
                            'Levels': {'L': 'left', 'R': 'right'}},        
                    'weight': {'Description': 'Body weight of the participant', 
                            'Unit': 'kg'},
                    'height': {'Description': 'Body height of the participant', 
                            'Unit': 'm'}                
                    }
        
        # Get a BIDS compatible path and filename
        filename = bids_path['root'] + '/' + 'participants.json'

        # Store the metadata in a json file
        with open(filename, 'w') as f:
            json.dump(metadata, f)
    
    return()

def make_dataset_description_json(bids_path, metadata):
    """
    Generate a a BIDS compatible dataset_description.tsv file

    Args:
        bids_path (dict): filename and filepath information
        subject_metadata (dict): metadata with essential keys
            - Name (str) 
            - BIDSversion (str)

    """

    # Check if the essential keys exist
    essentials = ['Name', 'BIDSversion']
    
    for i in np.arange(len(essentials)):
        if essentials[i] not in metadata:
            raise ValueError('essential keys are missing')

    # Get a BIDS compatible path and filename
    filename = bids_path['root'] + '/' + 'dataset_description.json'

    # Store the metadata in a json file
    with open(filename, 'w') as f:
        json.dump(metadata, f)

    return()

def make_dataset_readme():
    # ToDo

    return()

# Todo: Include CITATION.cff?

def raw_to_edf(data, fsamp, ch_names):
    edf_data = []
    return(edf_data)

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

    # Get a BIDS compatible path and filename
    path = bids_path['datapath']  
    name = bids_path['subject'] + '_' + bids_path['task'] + '_' + bids_path['datatype']

    edf.write(path + name + '.edf')
    return()

def make_bids_path(subject, task, datatype, root, session = -1, id_options=2):
    """
    Generate a a BIDS compatible folder structure

    Args:
            - subject (int): Unique subject ID
            - session (int): Session ID (otional)
            - task (str): Task description
            - datatype (str): Recorded modality (e.g. emg)
            - root (str): Root directory containing the dataset

    Returns:
        bids_path (dict): Dictonary containing root (str) and datapath (str)
    """        

    # Check if the function arguments are valid
    if type(subject) is not int or subject > 10**id_options-1:
        raise ValueError('invlaid subject ID')
    
    if type(session) is not int or session > 10**id_options-1:
        raise ValueError('invlaid session ID')
    
    if datatype not in ['emg', 'eeg', 'ieeg', 'meg']:
        raise ValueError('invalid datatype')

    # Process name and session input
    if session < 0:
        sub_name = 'sub' + '-' + str(subject).zfill(id_options)
        datapath = root + '/' + sub_name + '/' + datatype + '/'
    else:
        sub_name = 'sub' + '-' + str(subject).zfill(id_options)
        ses_name = 'ses' + '-' + str(session).zfill(id_options)
        datapath = root + '/' + sub_name + '/' + ses_name + '/' + datatype + '/'
    
    # Store essential information for BIDS compatible folder structure in a dictonary
    bids_path = {'root': root,
                 'datapath': datapath, 
                 'task': 'task-' + task,
                 'subject': sub_name,
                 'datatype': datatype
                 }
    
    # Generate an empty set of folders for hosting your BIDS dataset
    if not os.path.exists(datapath):
        os.makedirs(datapath)  

    return(bids_path)



import numpy as np
import os
import json
import pandas as pd
from edfio import *

class emg_bids_io:

    def __init__(self, 
                 subject=1, 
                 task = 'isometric', 
                 datatype = 'emg', 
                 datasetname = 'dataset-name',
                 root = './', 
                 session = -1, 
                 n_digits = 2):
      
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**n_digits-1:
            raise ValueError('invlaid subject ID')
        
        if type(session) is not int or session > 10**n_digits-1:
            raise ValueError('invlaid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        sub_name = 'sub' + '-' + str(subject).zfill(n_digits)
        if session < 0:
            datapath = root + datasetname + '/' + sub_name + '/' + datatype + '/'
        else:
            ses_name = 'ses' + '-' + str(session).zfill(n_digits)
            datapath = root + datasetname + '/' + sub_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.root = root + datasetname
        self.datapath = datapath
        self.task = 'task-' + task
        self.subject_id = sub_name
        self.datatype = datatype
        self.data = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.channels = pd.DataFrame(columns=['name', 'type', 'unit'])
        self.electrodes = pd.DataFrame(columns=['name','x','y','z', 'coordinate_system'])
        self.subject = pd.DataFrame(columns=['name', 'age', 'sex', 'hand', 'weight', 'height'])
        self.emg_sidecar = {'EMGPlacementScheme': [], 'EMGReference': [], 'SamplingFrequency': [],
                    'PowerLineFrequency': [], 'SoftwareFilters': [], 'TaskName': []}
        self.coord_sidecar = {'EMGCoordinateSystem': [], 'EMGCoordinateUnits': []}
        self.dataset_sidecar = {'Name': datasetname, 'BIDSversion': 'unpublished'}
        self.subject_sidecar = {'name': []} 

  

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        # write *_channels.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'channels' 
        self.channels.to_csv(name + '.tsv', sep='\t', index=False, header=True)
        # write *_electrode.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'electrodes' 
        self.electrodes.to_csv(name + '.tsv', sep='\t', index=False, header=True)
        # write *_emg.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype
        with open(name + '.json', 'w') as f:
            json.dump(self.emg_sidecar, f)
        # write *_coordsystem.json     
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'coordsystem'
        with open(name + '.json', 'w') as f:
            json.dump(self.coord_sidecar, f)
        # write participant.tsv
        name = self.root + '/' + 'participants.tsv'
        self.subject.to_csv(name, sep='\t', index=False, header=True)
        # write participant.json
        name = self.root + '/' + 'participants.json'
        with open(name, 'w') as f:
            json.dump(self.subject_sidecar, f)  
        # write dataset.json
        name = self.root + '/' + 'dataset.json'
        with open(name, 'w') as f:
            json.dump(self.dataset_sidecar, f) 
        # write edf file 
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype
        self.data.write(name + '.edf')  

        return()

    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_channels.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'channels.tsv' 
        if os.path.isfile(name):
            self.channels = pd.read_table(name)
        # read *_electrodes.tsv    
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'electrodes.tsv'
        if os.path.isfile(name):
            self.electrodes = pd.read_table(name)  
        # read *_emg.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype + '.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.emg_sidecar = json.load(f)
        # read *_coordsystem.json     
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'coordsystem.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.coord_sidecar = json.load(f)
        # read participant.tsv
        name = self.root + '/' + 'participants.tsv'
        if os.path.isfile(name):
            self.subject = pd.read_table(name)
        # read participant.json
        name = self.root + '/' + 'participants.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.subject_sidecar = json.load(f) 
        # read dataset.json
        name = self.root + '/' + 'dataset.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.dataset_sidecar = json.load(f) 
        # read edf file
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype + '.edf'
        if os.path.isfile(name):
            self.data = read_edf(name)      
              
        return()  
                      

    def add_channel_metadata(self, new_metadata):
        """
        Add channel metadata

        Args:
            new_metadata (dict or pd.core.frame.DataFrame): Metadata

            List of essential keys (ordering matters):
                - name (string)
                - type (string)
                - units (string)

        """
    
        # If an electrode already exists overwrite existing metadata
        if type(new_metadata) == dict:
            df_new = pd.DataFrame(data=new_metadata)
            frames = [self.channels, df_new]
        elif type(new_metadata) == pd.core.frame.DataFramect:
            frames = [self.channels, new_metadata]
        else:
            raise ValueError('input has incorrect datatype')
        
        self.channels = pd.concat(frames)
        self.channels = self.channels.drop_duplicates(subset='name', keep='last')

        return()

    def add_electrode_metadata(self, new_metadata):
        """
        Add electrode metadata

        Args:
            new_metadata (dict or pd.core.frame.DataFrame): Metadata

            List of essential keys (ordering matters): 
                - name (string)
                - x (float)
                - y (float)
                - z (float)
                - coordinate_system (string)

        """

        # If an electrode already exists overwrite existing metadata
        if type(new_metadata) == dict:
            df_new = pd.DataFrame(data=new_metadata)
            frames = [self.electrodes, df_new]
        elif type(new_metadata) == pd.core.frame.DataFramect:
            frames = [self.electrodes, new_metadata]
        else:
            raise ValueError('input has incorrect datatype')
        
        self.electrodes = pd.concat(frames)
        self.electrodes = self.subject.drop_duplicates(subset=['name', 'coordinate_system'], keep='last')

        # If the z coordinate exist make sure the ordering is correct
        if 'z' in self.electrodes.columns:
            col = self.electrodes.pop('z')
            self.electrodes.insert(3, 'z', col)    

        return()

    def add_emg_sidecar_metadata(self, new_metadata):
        """
        Update metadata of the emg sidecar file 

        Args:
            new_metadata (dict or path): metadata 

            List of essential keys: 
                - EMGPlacementScheme (str)
                - EMGReference (str)
                - SamplingFrequency (float)
                - PowerLineFrequency (float or "n/a")
                - SoftwareFilters (dict or "n/a")
                - TaskName (str)

        """

        if type(new_metadata) == dict:
            self.emg_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.emg_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')    

        return()

    def add_coordsystem_sidecar_metadata(self, new_metadata):
        """
        Add metadata that should be stored in coordsystem.json 

        Args:
            new_metadata (dict or path): metadata 

            List of essential keys:
                - EMGCoordinateSystem (str)
                - EMGCoordinateUnits (str)

        """

        if type(new_metadata) == dict:
            self.coord_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.coord_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')      

        return()

    def add_subject_metadata(self, subject_metadata):
        """
        Add new subject 

        Args:
            subject_metadata (dict): metadata with essential keys
                - name (str) 
                - age (float or "n/a")
                - sex (str)
                - hand (str)
                - weight (float or "n/a")
                - height (float or "n/a") 
        """

        # If a subject already exist only overwrite existing metadata
        df_new = pd.DataFrame(data=subject_metadata, index=[0])
        frames = [self.subject, df_new]
        self.subject = pd.concat(frames)
        self.subject = self.subject.drop_duplicates(subset='name', keep='last')

        return()

    def add_participant_sidecar_metadata_from_template(self,data_type):
        """
        Add metadata for a a BIDS compatible participants.json file from a predined template

        Args:
            data_type (str): select from predefined template 'simulation' or 'experimental'

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
            
            self.subject_sidecar.update(metadata)
        
        return()

    def add_dataset_sidecar_metadata(self, new_metadata):
        """
        Add metadata to dataset_sidecar 

        Args:
            new_metadata (dict or path): metadata

            List of essential keys
                - Name (str) 
                - BIDSversion (str)

        """

        if type(new_metadata) == dict:
            self.dataset_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.dataset_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')

        return()

    def make_dataset_readme():
        # ToDo

        return()
    
    def set_raw_data(self, mydata, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            data (np.ndarry): emg_data (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds 
        seconds = np.ceil(mydata.shape[0]/fsamp)
        signal = np.zeros([int(seconds*fsamp), mydata.shape[1]])
        signal[0:mydata.shape[0],:] = mydata

        # Initalize
        edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp)])

        for i in np.arange(1,signal.shape[1]):
            new_signal = EdfSignal(signal[:,i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)

        self.data = edf

        return()
    
    def edf_to_numpy(self, idx):
        """
        Output data of selcetd channels as numpy array

        Args:
            idx (ndarray): Indices of the channels to be stored

        """

        data_out = np.zeros((self.data.signals[idx[0]].data.shape[0], len(idx)))
        for i in np.arange(len(idx)):
            data_out[:,i] = self.data.signals[idx[i]].data

        return(data_out)  


class decomp_derivatives_bids_io:

    def __init__(self, 
                 subject=1, 
                 task = 'isometric', 
                 datatype = 'emg', 
                 datasetname = 'dataset-name',
                 pipelinename = 'pipeline-name',
                 root = './', 
                 session = -1, 
                 n_digits = 2):
      
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**n_digits-1:
            raise ValueError('invlaid subject ID')
        
        if type(session) is not int or session > 10**n_digits-1:
            raise ValueError('invlaid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        sub_name = 'sub' + '-' + str(subject).zfill(n_digits)
        if session < 0:
            datapath = sub_name + '/' + datatype + '/'
        else:
            ses_name = 'ses' + '-' + str(session).zfill(n_digits)
            datapath = sub_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.pipelinename = pipelinename
        self.root = root + datasetname + '_' + pipelinename
        self.datapath = self.root + '/' + datapath
        self.task = 'task-' + task
        self.subject_id = sub_name
        self.datatype = datatype
        self.source = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.spikes = pd.DataFrame(columns=['source_id', 'spike_time'])
        self.pipeline_sidecar = {'PipelineName': pipelinename, 
                                 'PipelineParameters': [],
                                 'PipelineDescription': [], 
                                 'SamplingFrequency': []}
        self.dataset_sidecar = {'Name': datasetname + '_' + pipelinename, 
                                'BIDSversion': 'unpublished', 
                                'GeneratedBy': pipelinename}
        

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        # write *predictedspikes.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedspikes' 
        self.spikes.to_csv(name + '.tsv', sep='\t', index=False, header=True)
        # write *_pipeline.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'pipeline'
        with open(name + '.json', 'w') as f:
            json.dump(self.pipeline_sidecar, f)
        # write dataset.json
        name = self.root + '/' + 'dataset.json'
        with open(name, 'w') as f:
            json.dump(self.dataset_sidecar, f) 
        # write edf file 
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedsources'
        self.source.write(name + '.edf') 

        return()
    
    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_predictedspikes.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedspikes.tsv' 
        if os.path.isfile(name):
            self.spikes = pd.read_table(name)
        # read *_pipeline.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'pipeline.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.pipeline_sidecar = json.load(f)
        # read dataset.json
        name = self.root + '/' + 'dataset.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.dataset_sidecar = json.load(f) 
        # read edf file
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedsources.edf'
        if os.path.isfile(name):
            self.source = read_edf(name)    
              
        return()
    
    def add_spikes(self,spikes):
        """
        Convert a dictionary of spike times to long-format TSV-style DataFrame.

        Parameters:
            spike_dict (dict): {source_id: list of spike times}

        """
        rows = []
        for unit_id, spike_times in spikes.items():
            for t in spike_times:
                rows.append({'source_id': unit_id, 'spike_time': t})

        frames = [self.spikes, pd.DataFrame(rows)]
        self.spikes = pd.concat(frames, ignore_index=True)
        self.spikes = self.spikes.drop_duplicates(subset=['source_id', 'spike_time'])

        return()
    
    def add_dataset_sidecar_metadata(self, new_metadata):
        """
        Add metadata to dataset_sidecar 

        Args:
            new_metadata (dict or path): metadata

            List of essential keys
                - Name (str) 
                - BIDSversion (str)
                - GeneratedBy (str)

        """

        if type(new_metadata) == dict:
            self.dataset_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.dataset_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')

        return()
    
    def add_dataset_sidecar_metadata(self, new_metadata):
        """
        Add metadata to pipeline_sidecar 

        Args:
            new_metadata (dict or path): metadata

            List of essential keys
                - ...
                - ...

        """

        if type(new_metadata) == dict:
            self.pipeline_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.pipeline_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')

        return()
    
    def set_source_data(self, mysources, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            data (np.ndarry): emg_data (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds 
        seconds = np.ceil(mysources.shape[0]/fsamp)
        signal = np.zeros([int(seconds*fsamp), mysources.shape[1]])
        signal[0:mysources.shape[0],:] = mysources

        # Initalize
        edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp)])

        for i in np.arange(1,signal.shape[1]):
            new_signal = EdfSignal(signal[:,i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)

        self.source = edf

        return()







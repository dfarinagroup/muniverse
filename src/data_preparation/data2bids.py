import numpy as np
import os
import json
import pandas as pd
from edfio import *

class emg_bids_generator:

    def __init__(self, subject, task, datatype, root, session = -1, id_options=2):
      
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**id_options-1:
            raise ValueError('invlaid subject ID')
        
        if type(session) is not int or session > 10**id_options-1:
            raise ValueError('invlaid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        if session < 0:
            sub_name = 'sub' + '-' + str(subject).zfill(id_options)
            datapath = root + '/' + sub_name + '/' + datatype + '/'
        else:
            sub_name = 'sub' + '-' + str(subject).zfill(id_options)
            ses_name = 'ses' + '-' + str(session).zfill(id_options)
            datapath = root + '/' + sub_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.root = root
        self.datapath = datapath
        self.task = 'task-' + task
        self.subject_id = sub_name
        self.datatype = datatype
        self.data = np.empty
        self.fsamp = float()
        self.channels = pd.DataFrame(columns=['name', 'type', 'unit'])
        self.electrodes = pd.DataFrame(columns=['name','x','y','z', 'coordinate_system'])
        self.subject = pd.DataFrame(columns=['name', 'age', 'sex', 'hand', 'weight', 'height'])
        self.emg_sidecar = {'EMGPlacementScheme': [], 'EMGReference': [], 'SamplingFrequency': [],
                    'PowerLineFrequency': [], 'SoftwareFilters': [], 'TaskName': []}
        self.coord_sidecar = {'EMGCoordinateSystem': [], 'EMGCoordinateUnits': []}
        self.dataset_sidecar = {'Name': [], 'BIDSversion': 'unpublished'}
        self.subject_sidecar = {'name': []} 

        # Generate an empty set of folders for hosting your BIDS dataset
        if not os.path.exists(datapath):
            os.makedirs(datapath)  

    def write(self):
        """
        Save dataset in BIDS format

        """

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

        # Convert data into edf format 
        seconds = np.ceil(self.data.shape[0]/self.fsamp)
        # Add zeros to the signal such that the total length is in full seconds
        signal = np.zeros([int(seconds*self.fsamp), self.data.shape[1]])
        signal[0:self.data.shape[0],:] = self.data

        edf = Edf([EdfSignal(signal[:,0], sampling_frequency=self.fsamp, 
                             label=self.channels.loc[0, 'name'],
                             physical_dimension=self.channels.loc[0, 'unit'])])

        for i in np.arange(1,signal.shape[1]):
            new_signal = EdfSignal(signal[:,i], 
                                   sampling_frequency=self.fsamp, 
                                   label=self.channels.loc[i, 'name'],
                                   physical_dimension=self.channels.loc[i, 'unit'])
            edf.append_signals(new_signal)

        # Get a BIDS compatible path and filename 
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype
        edf.write(name + '.edf')  

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
            edf = read_edf(name)
            self.fsamp = edf.signals[0].sampling_frequency
            self.data = np.zeros((edf.signals[0].data.shape[0], edf.num_signals))
            for i in np.arange(edf.num_signals):
                self.data[:,i] = edf.signals[i].data
  
        return()  
                      

    def add_channel_metadata(self, channel_metadata):
        """
        Add channel metadata

        Args:
            channel_metadata (dict): Channel metadata with essential keys
                - name (string)
                - type (string)
                - units (string)

        """
    
        # If an electrode already exists overwrite existing metadata
        df_new = pd.DataFrame(data=channel_metadata)
        frames = [self.channels, df_new]
        self.channels = pd.concat(frames)
        self.channels = self.channels.drop_duplicates(subset='name', keep='last')

        return()

    def add_electrode_metadata(self, el_metadata):
        """
        Add electrode metadata

        Args:
            el_metadata (dict): electrode metadata with essential keys
                - name (string)
                - x (float)
                - y (float)
                - z (float)
                - coordinate_system (string)

        """

        # If an electrode already exists overwrite existing metadata
        df_new = pd.DataFrame(data=el_metadata)
        frames = [self.electrodes, df_new]
        self.electrodes = pd.concat(frames)
        self.electrodes = self.subject.drop_duplicates(subset=['name', 'coordinate_system'], keep='last')

        # If the z coordinate exist set it at the right position
        if 'z' in self.electrodes.columns:
            col = self.electrodes.pop('z')
            self.electrodes.insert(3, 'z', col)    

        return()

    def set_emg_sidecar(self, emg_metadata):
        """
        Update metadata of the emg sidecar file 

        Args:
            emg_metadata (dict): metadata with essential keys
                - EMGPlacementScheme (str)
                - EMGReference (str)
                - SamplingFrequency (float)
                - PowerLineFrequency (float or "n/a")
                - SoftwareFilters (dict or "n/a")
                - TaskName (str)

        """

        self.emg_sidecar.update(emg_metadata)

        return()

    def set_coordsystem_sidecar(self, coordsystem_metadata):
        """
        Add metadata that should be stored in coordsystem.json 

        Args:
            bids_path (dict): filename and filepath information
            coordsystem_metadata (dict): metadata with essential keys
                - EMGCoordinateSystem (str)
                - EMGCoordinateUnits (str)

        """

        self.coord_sidecar.update(coordsystem_metadata)

        return()

    def add_subject(self, subject_metadata):
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

    def set_participant_sidecar_temp(self,data_type):
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

    def set_dataset_sidecar(self, metadata):
        """
        Add metadata to dataset_sidecar 

        Args:
            metadata (dict): metadata with essential keys
                - Name (str) 
                - BIDSversion (str)

        """

        self.dataset_sidecar.update(metadata)

        return()

    def make_dataset_readme():
        # ToDo

        return()
    
    def set_raw_data(self, data, fsamp):
        """
        Add raw data

        Args:
            data (np.ndarry): emg_data (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        self.data = data
        self.fsamp = fsamp

        return()





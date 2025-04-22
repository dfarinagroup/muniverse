import numpy as np
import os
import json
import pandas as pd
from edfio import *
import h5py
    
class bids_dataset:

    def __init__(self,
                 datasetname = 'dataset-name',
                 root = './',
                 n_digits = 2, 
                 overwrite = False):
        
        self.root = root + datasetname
        self.datasetname = datasetname
        self.dataset_sidecar = {'Name': datasetname, 'BIDSversion': self._get_bids_version()}
        self.subjects_sidecar = self._set_participant_sidecar()
        self.subjects_data = pd.DataFrame(columns=['name', 'age', 'sex', 'hand', 'weight', 'height'])
        self.n_digits = n_digits
        self.overwrite = overwrite

    def write(self):
        """
        Export BIDS dataset

        """

        # make folder
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # write participant.tsv
        name = self.root + '/' + 'participants.tsv'
        if self.overwrite or not os.path.isfile(name):  
            self.subjects_data.to_csv(name, sep='\t', index=False, header=True)
        elif not self.overwrite and os.path.isfile(name):
            from_file = pd.read_table(name)
            if not from_file.equals(self.subjects_data):
                self.subjects_data.to_csv(name, sep='\t', index=False, header=True)
        # write participant.json
        name = self.root + '/' + 'participants.json'
        if self.overwrite or not os.path.isfile(name): 
            with open(name, 'w') as f:
                json.dump(self.subjects_sidecar, f)  
        # write dataset.json
        name = self.root + '/' + 'dataset.json'
        if self.overwrite or not os.path.isfile(name):
            with open(name, 'w') as f:
                json.dump(self.dataset_sidecar, f)

        return()
            
    def read(self):
        """
        Import data from BIDS dataset

        """

        # read participant.tsv
        name = self.root + '/' + 'participants.tsv'
        if os.path.isfile(name):
            self.subjects_data = pd.read_table(name)
        # read participant.json
        name = self.root + '/' + 'participants.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.subjects_sidecar = json.load(f) 
        # read dataset.json
        name = self.root + '/' + 'dataset.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.dataset_sidecar = json.load(f) 

        return()

    def set_metadata(self, field_name, source):
        """
        Generic metadata update function.

        Parameters:
            field_name (str): name of the metadata attribute to update
            source (dict, DataFrame, or str): data or file path
        """
        current = getattr(self, field_name, None)
        if current is None:
            raise ValueError(f"No such field '{field_name}'")

        # Load from file if needed
        if isinstance(source, str):
            if source.endswith(".json"):
                with open(source) as f:
                    source = json.load(f)
            elif source.endswith(".tsv"):
                source = pd.read_csv(source, sep="\t")
            else:
                raise ValueError(f"Unsupported file type: {source}")

        # Update logic based on current type
        if isinstance(current, dict):
            if isinstance(source, dict):
                current.update(source)
            elif isinstance(source, pd.DataFrame):
                current.update(source.to_dict(orient='records')[0])  # assumes one row
            else:
                raise TypeError("Expected dict or DataFrame for dict field")
        elif isinstance(current, pd.DataFrame):
            if isinstance(source, dict):
                row = pd.DataFrame(data=source)
                current = pd.concat([current, row], ignore_index=True)
            elif isinstance(source, pd.DataFrame):
                current = pd.concat([current, source], ignore_index=True)
            else:
                raise TypeError("Expected dict or DataFrame for DataFrame field")
        else:
            raise TypeError(f"Unsupported target type for '{field_name}'")
        
        if field_name == 'subjects_data':
            current.drop_duplicates(subset='name', keep='last')
            current.sort_values('name')

        # Update field 
        setattr(self, field_name, current)

    def _set_participant_sidecar(self):
        '''
        Return a template for initalizing the participant sidecar file
        
        '''

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

        return metadata   

    def _get_bids_version(self):
        '''
        Get the BIDS version of your dataset

        '''

        bids_version = 'extension proposal for electromyography (EMG) - BEP042'

        return bids_version         

                

class bids_emg_recording(bids_dataset):

    def __init__(self, 
                subject=1, 
                task = 'isometric', 
                datatype = 'emg', 
                session = -1, 
                run = 1,
                data_obj = None,
                root = './',
                datasetname = 'my-data',
                overwrite = False,
                n_digits = 2):
            
        super().__init__(
            datasetname = datasetname,
            root = root, 
            overwrite = overwrite,
            )
        
        if isinstance(data_obj, bids_dataset):
            self.root = data_obj.root
            self.datasetname = data_obj.datasetname
            
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**n_digits-1:
            raise ValueError('invalid subject ID')
        
        if type(session) is not int or session > 10**n_digits-1:
            raise ValueError('invalid session ID')
        
        if type(run) is not int or run > 10**n_digits-1:
            raise ValueError('invalid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        subject_name = 'sub' + '-' + str(subject).zfill(n_digits)
        if session < 0:
            datapath = self.root + '/' + subject_name + '/' + datatype + '/'
        else:
            ses_name = 'ses' + '-' + str(session).zfill(n_digits)
            datapath = self.root + '/' + subject_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        self.n_digits = n_digits
        self.subject_id = subject
        self.subject_name = subject_name
        self.task = 'task-' + task
        self.run = run
        self.datatype = datatype
        self.emg_data = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.channels = pd.DataFrame(columns=['name', 'type', 'unit'])
        self.electrodes = pd.DataFrame(columns=['name','x','y','z', 'coordinate_system'])
        self.emg_sidecar = {'EMGPlacementScheme': [], 'EMGReference': [], 'SamplingFrequency': [],
                    'PowerLineFrequency': [], 'SoftwareFilters': [], 'TaskName': []}
        self.coord_sidecar = {'EMGCoordinateSystem': [], 'EMGCoordinateUnits': []} 

    def write(self):
        """
        Save dataset in BIDS format

        """

        super().write()
        
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        # Make BIDS compatible file names  
        name = self.datapath + self.subject_name + '_' + self.task + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # write *_channels.tsv
        self.channels.to_csv(name + 'channels.tsv', sep='\t', index=False, header=True)
        # write *_electrode.tsv
        self.electrodes.to_csv(name + 'electrodes.tsv', sep='\t', index=False, header=True)
        # write *_emg.json  
        with open(name + self.datatype + '.json', 'w') as f:
            json.dump(self.emg_sidecar, f)
        # write *_coordsystem.json     
        with open(name + 'coordsystem.json', 'w') as f:
            json.dump(self.coord_sidecar, f) 
        # write edf file 
        self.emg_data.write(name + self.datatype +  '.edf') 


    def read(self):
        """
        Import data from BIDS dataset

        """
        super().read()

        # Make BIDS compatible file names   
        name = self.datapath + self.subject_name + '_' + self.task + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # read *_channels.tsv 
        if os.path.isfile(name + 'channels.tsv'):
            self.channels = pd.read_table(name + 'channels.tsv')
        # read *_electrodes.tsv    
        if os.path.isfile(name + 'electrodes.tsv'):
            self.electrodes = pd.read_table(name + 'electrodes.tsv')  
        # read *_emg.json  
        if os.path.isfile(name + self.datatype + '.json'):
            with open(name + self.datatype + '.json', 'r') as f:
                self.emg_sidecar = json.load(f)
        # read *_coordsystem.json     
        if os.path.isfile(name + 'coordsystem.json'):
            with open(name + 'coordsystem.json', 'r') as f:
                self.coord_sidecar = json.load(f)
        # read edf file
        if os.path.isfile(name + self.datatype + '.edf'):
            self.emg_data = read_edf(name + self.datatype + '.edf')

    def set_metadata(self, field_name, source):

        super().set_metadata(field_name, source)

        if field_name == 'channels':
            # Drop duplicates
            self.channels = self.channels.drop_duplicates(subset='name', keep='last')
        elif field_name == 'electrodes':
            # Drop duplicates
            self.electrodes = self.electrodes.drop_duplicates(subset=['name', 'coordinate_system'], keep='last')
            # If the z coordinate exist make sure the ordering is correct
            if 'z' in self.electrodes.columns:
                col = self.electrodes.pop('z')
                self.electrodes.insert(3, 'z', col)

    def set_data(self, field_name, mydata, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            field_name (str): name of the field to be updated 
            mydata (np.ndarry): data matrix (n_samples x n_channels)
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

        # Set data 
        setattr(self, field_name, edf)

        return()                  



class simulated_emg_bids_io(bids_emg_recording):
    def __init__(self, config_path, hdf5_path, root='./data', datasetname='simulated-dataset'):
        # Parse config file first
        self.config = self._parse_config(config_path)
        self.hdf5_path = hdf5_path
        
        # Initialize the parent class with BIDS-compatible parameters
        subject_id = self._generate_subject_id()
        task_name = self._generate_task_name()
        
        super().__init__(
            subject=subject_id,
            task=task_name,
            datatype='emg',
            datasetname=datasetname,
            root=root
        )
        
        # Initialize additional data structures for simulated data
        self.spikes = pd.DataFrame(columns=['source_id', 'spike_time'])
        self.motor_units = pd.DataFrame(columns=['unit_id', 'recruitment_threshold', 'depth', 'innervation_zone', 
                                               'fibre_density', 'fibre_length', 'conduction_velocity', 'angle'])
        self.internals = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        # self.spikes_sidecar = {'source_id': [], 'spike_time': []}
        # self.motor_units_sidecar = {'unit_id': [], 'recruitment_threshold': [], 'depth': [], 'innervation_zone': [], 'fibre_density': [], 'fibre_length': [], 'conduction_velocity': [], 'angle':[]}
        self.internals_sidecar = {'SamplingFrequency': [], 'Description': 'Simulated internal variables'}
        
    def _parse_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def _generate_subject_id(self):
        return f"sim{self.config['SubjectConfiguration']['SubjectSeed']}"
        
    def _generate_task_name(self):
        # Generate a BIDS-compatible task name
        movement = self.config['MovementConfiguration']
        profile = movement['MovementProfileParameters']
        muscle = movement['TargetMuscle']
        movement_type = movement['MovementType']
        effort_level = profile['EffortLevel']
        if movement_type == 'Isometric':
            effort_profile = profile['EffortProfile']
        else:
            effort_profile = profile['AngleProfile']
        
        # Example: "FDSIisometrictrapezoid10pct"
        return f"{muscle.upper()}{movement_type.lower()}{effort_profile.lower()}{effort_level}pct"
        
    def _setup_electrode_metadata(self):
        # Generate electrode metadata based on RecordingConfiguration
        elec_config = self.config['RecordingConfiguration']['ElectrodeConfiguration']
        
        # Generate electrode names and positions: TO-DO
        names = []
        x_pos = []
        y_pos = []
        coordinate_system = []
        
        for i in range(elec_config['NElectrodes']):
            row = i // elec_config['NCols']
            col = i % elec_config['NCols']
            
            names.append(f"E{i+1}")
            x_pos.append(col * elec_config['InterElectrodeDistance'])
            y_pos.append(row * elec_config['InterElectrodeDistance'])
            coordinate_system.append("Grid1")
            
        return {
            'name': names,
            'x': x_pos,
            'y': y_pos,
            'coordinate_system': coordinate_system
        }
        
    def _setup_channel_metadata(self):
        # Generate channel metadata
        elec_config = self.config['RecordingConfiguration']['ElectrodeConfiguration']
        
        names = []
        types = []
        units = []
        
        for i in range(elec_config['NElectrodes']):
            names.append(f"E{i+1}")
            types.append("EMG")
            units.append("mV")
            
        return {
            'name': names,
            'type': types,
            'unit': units
        }
        
    def _setup_coordsystem_metadata(self):
        return {
            'EMGCoordinateSystem': 'local',
            'EMGCoordinateUnits': 'mm'
        }
        
    def _setup_spikes_data(self):
        """Read and format spike times from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            spike_times = f['spikes'][:]
            unit_ids = f['unit_ids'][:]
            
        # Convert to long format DataFrame
        rows = []
        for unit_id, times in zip(unit_ids, spike_times):
            for t in times:
                rows.append({'source_id': unit_id, 'spike_time': t})
                
        self.spikes = pd.DataFrame(rows)
        
    def _setup_motor_units_data(self):
        """Read and format motor unit properties from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            unit_ids = f['unit_ids'][:]
            recruitment_thresholds = f['recruitment_thresholds'][:]
            unit_properties = f['unit_properties'][:]

        # Handle individual properties: TO-DO
        self.motor_units = pd.DataFrame({
            'unit_id': unit_ids,
            'recruitment_threshold': recruitment_thresholds,
            'depth': unit_properties[:,0],
            'innervation_zone': unit_properties[:,1],
            'fibre_density': unit_properties[:,2],
            'fibre_length': unit_properties[:,3],
            'conduction_velocity': unit_properties[:,4],
            'angle': unit_properties[:,5]
        })
        
    def _setup_internals_data(self):
        """Read and format internal simulation variables from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            internals_data = f['_internals'][:]
            fsamp = self.config['RecordingConfiguration']['SamplingFrequency']
            
        # Convert to EDF format
        edf = Edf([EdfSignal(internals_data[:,0], sampling_frequency=fsamp)])
        for i in range(1, internals_data.shape[1]):
            new_signal = EdfSignal(internals_data[:,i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)
            
        self.internals = edf
        self.internals_sidecar['SamplingFrequency'] = fsamp
        
    def _write_simulated_files(self):
        """Write additional simulated data files"""
        # Write spikes data
        spikes_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_spikes")
        self.spikes.to_csv(spikes_name + '.tsv', sep='\t', index=False)
        
        # Write motor units data
        mu_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_motorunits")
        self.motor_units.to_csv(mu_name + '.tsv', sep='\t', index=False)
        
        # Write internals data
        internals_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_internals")
        self.internals.write(internals_name + '.edf')
        with open(internals_name + '.json', 'w') as f:
            json.dump(self.internals_sidecar, f)
            
    def write(self):
        """Override write method to include simulated data"""
        # First set up all the simulated data
        self._setup_spikes_data()
        self._setup_motor_units_data()
        self._setup_internals_data()
        
        # Call parent's write method to handle standard BIDS files
        super().write()
        
        # Write additional simulated files
        self._write_simulated_files()
        
    def read(self):
        """Override read method to include simulated data"""
        # Call parent's read method first
        super().read()
        
        # Read simulated data files if they exist
        spikes_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_spikes.tsv")
        if os.path.isfile(spikes_name):
            self.spikes = pd.read_table(spikes_name)
            
        mu_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_motorunits.tsv")
        if os.path.isfile(mu_name):
            self.motor_units = pd.read_table(mu_name)
            
        internals_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_internals.edf")
        if os.path.isfile(internals_name):
            self.internals = read_edf(internals_name)
            with open(internals_name.replace('.edf', '.json'), 'r') as f:
                self.internals_sidecar = json.load(f)


class bids_decomp_derivatives(bids_emg_recording):

    def __init__(self, 
                 pipelinename = 'pipeline-name',
                 format = 'standalone',
                 data_obj = None, 
                 datasetname = 'dataset-name',
                 datatype = 'emg',
                 subject=1, 
                 task = 'isometric', 
                 run = 1,
                 session = -1, 
                 root = './',
                 overwrite = False,              
                 n_digits = 2):
      
        super().__init__(
            subject=subject, 
            task = task, 
            datatype = datatype, 
            session = session, 
            run = run,
            data_obj = None,
            root = root,
            datasetname = datasetname,
            overwrite = overwrite,
            n_digits = n_digits
        )

        if isinstance(data_obj, bids_emg_recording):
            self.root = data_obj.root
            self.datasetname = data_obj.datasetname
            self.datapath = data_obj.datapath
            self.n_digits = data_obj.n_digits
            self.subject_id = data_obj.subject_id
            self.subject_name = data_obj.subject_name
            self.task = data_obj.task
            self.run = data_obj.task
            self.datatype = data_obj.datatype
            self.emg_data = data_obj.emg_data
            self.channels = data_obj.channels
            self.electrodes = data_obj.electrodes
            self.emg_sidecar = data_obj.emg_sidecar
            self.coord_sidecar = data_obj.coord_sidecar
            self.dataset_sidecar = data_obj.dataset_sidecar
            self.subjects_data = data_obj.subjects_data
            self.subjects_sidecar = data_obj.subjects_sidecar


        # Store essential information for BIDS compatible folder structure in a dictonary
        if format == 'standalone':
            self.datasetname = self.datasetname + '-' + pipelinename
            self.derivative_root = root + self.datasetname + '/'
            self.derivative_datapath = self.derivative_root + self.subject_name + '/' + self.datatype + '/' 
        else:
            self.derivative_root = self.root + + 'derivatives/' + pipelinename + '/'
            self.derivative_datapath = self.derivative_root + self.subject_name + './' + self.datatype + '/' 

        self.pipelinename = pipelinename

        self.source = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.spikes = pd.DataFrame(columns=['source_id', 'spike_time'])
        self.pipeline_sidecar = {'PipelineName': pipelinename, 
                                 'PipelineParameters': [],
                                 'PipelineDescription': [], 
                                 'SamplingFrequency': []}
        self.dataset_sidecar = {'Name': datasetname + '_' + pipelinename, 
                                'BIDSversion': self._get_bids_version(), 
                                'GeneratedBy': pipelinename}
        

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.derivative_datapath):
            os.makedirs(self.derivative_datapath)

        name = self.derivative_datapath + self.subject_name + '_' + self.task + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # write *_predictedspikes.tsv
        self.spikes.to_csv(name + 'predictedspikes.tsv', sep='\t', index=False, header=True)
        # write *_pipeline.json 
        fname = name + 'pipeline.json' 
        with open(fname, 'w') as f:
            json.dump(self.pipeline_sidecar, f)
        # write *_predictedsources.edf file 
        self.source.write(name + 'predictedsources.edf')     
        # write dataset.json
        fname = self.derivative_root + '/' + 'dataset.json'
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, 'w') as f:
                json.dump(self.dataset_sidecar, f) 

        
    
    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_predictedspikes.tsv
        name = self.derivative_datapath + self.subject_name + '_' + self.task + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # read *_predictedspikes.tsv
        fname = name + 'predictedspikes.tsv' 
        if os.path.isfile(fname):
            self.spikes = pd.read_table(fname)
        # read *_pipeline.json  
        fname = name + 'pipeline.json'
        if os.path.isfile(fname):
            with open(fname, 'r') as f:
                self.pipeline_sidecar = json.load(f)
        # read *.edf file
        fname = name + 'predictedsources.edf'
        if os.path.isfile(fname):
            self.source = read_edf(fname)         
        # read dataset.json
        fname = self.derivative_root + '/' + 'dataset.json'
        if os.path.isfile(fname):
            with open(fname, 'r') as f:
                self.dataset_sidecar = json.load(f) 

           
    
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

    
    
def edf_to_numpy(edf_data, idx):
    """
    Output data of selcetd channels as numpy array

    Args:
        edf_data (edf): Time series data in edf format
        idx (ndarray): Indices of the channels to be stored

    Returns:
        np_data (np.ndarray): Time series data  
    """

    np_data = np.zeros((edf_data.signals[idx[0]].data.shape[0], len(idx)))
    for i in np.arange(len(idx)):
        np_data[:,i] = edf_data.signals[idx[i]].data

    return np_data






import numpy as np
import os
import json
import pandas as pd
from edfio import Edf, EdfSignal, read_edf
    
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
        name = self.root + '/' + 'dataset_description.json'
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
        name = self.root + '/' + 'dataset_description.json'
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

        bids_version = 'extension proposal for electromyography - BEP042'

        return bids_version         

                

class bids_emg_recording(bids_dataset):
    """
    Class for handling EMG recordings in BIDS format.
    
    This class implements the BIDS standard for EMG data (BEP042), including support for
    session-level inheritance of metadata files. By default, all metadata files are linked
    to their respective recording files. However, certain metadata files can be inherited
    at the session level to avoid duplication.
    
    Inheritance Rules:
    - By default, no metadata files are inherited (all are linked to _emg.edf)
    - Only electrodes.tsv and coordsystem.json can be inherited at session level
    - Inherited files are stored at session level with names like:
      sub-01_ses-01_electrodes.tsv
    - Non-inherited files are stored with recording files like:
      sub-01_ses-01_task-rest_run-01_electrodes.tsv
    """

    # Define valid metadata files that can be inherited
    # BEP042 identifies electrodes.tsv and coordsystem.json as candidates for session level inheritance
    # But by default, no metadata files are inherited (i.e., all are linked to the _emg.edf recording file)
    INHERITABLE_FILES = ['electrodes', 'coordsystem']

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
                n_digits = 2,
                inherited_metadata = None):
            
        super().__init__(
            datasetname = datasetname,
            root = root, 
            overwrite = overwrite,
            )
        
        if isinstance(data_obj, bids_dataset):
            self.root = data_obj.root
            self.datasetname = data_obj.datasetname
            self.subjects_data = data_obj.subjects_data
            
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
        self.session = session
        self.run = run
        self.datatype = datatype
        self.emg_data = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        
        # Initialize metadata
        self.channels = pd.DataFrame(columns=['name', 'type', 'unit'])
        self.electrodes = pd.DataFrame(columns=['name','x','y','z', 'coordinate_system'])
        self.emg_sidecar = {'EMGPlacementScheme': [], 'EMGReference': [], 'SamplingFrequency': [],
                    'PowerLineFrequency': [], 'SoftwareFilters': [], 'TaskName': []}
        self.coord_sidecar = {'EMGCoordinateSystem': [], 'EMGCoordinateUnits': []}
        
        # Initialize empty inheritance dictionary
        self.inherited_metadata = {}
        
        # Set inherited metadata if provided
        if inherited_metadata is not None:
            self.set_inherited_metadata(inherited_metadata)

    def set_inherited_metadata(self, metadata_files):
        """
        Set which metadata files should be inherited at session level.
        
        Parameters:
        -----------
        metadata_files : list of str
            List of metadata file names to inherit. Must be from INHERITABLE_FILES.
            Example: ['electrodes', 'coordsystem']
            
        Notes:
        ------
        - Only electrodes.tsv and coordsystem.json can be inherited
        - Inherited files are stored at session level
        - Non-inherited files are stored with their respective recording files
        """
        # Validate input
        invalid_files = [f for f in metadata_files if f not in self.INHERITABLE_FILES]
        if invalid_files:
            raise ValueError(f"Invalid metadata files for inheritance: {invalid_files}. "
                           f"Valid options are: {self.INHERITABLE_FILES}")
        
        # Set inheritance flags
        self.inherited_metadata = {f: True for f in metadata_files}

    def _get_session_path(self):
        """Get the session-level path for inherited files"""
        if self.session < 0:
            return self.datapath
        return os.path.dirname(os.path.dirname(self.datapath)) + '/'

    def _get_session_prefix(self):
        """Get the session-level prefix for inherited files"""
        return self._get_session_path() + self.subject_name + '_' + f'ses-{str(self.session).zfill(self.n_digits)}'

    def _get_metadata_filename(self, metadata_type):
        """Get the appropriate filename for a metadata file based on inheritance"""
        if self.inherited_metadata.get(metadata_type, False) and self.session > 0:
            return self._get_session_prefix() + f'_{metadata_type}'
        else:
            name = self.datapath + self.subject_name + '_' + self.task + '_'
            if self.session > 0:
                name = name + 'ses-' + str(int(self.session)).zfill(self.n_digits) + '_'
            if self.run > 0:
                name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'
            return name + metadata_type

    def _write_metadata_file(self, metadata_type, data, writer_func):
        """
        Write a metadata file, handling inheritance appropriately.
        
        Parameters:
        -----------
        metadata_type : str
            Type of metadata file (e.g., 'electrodes', 'coordsystem')
        data : object
            Data to write (DataFrame or dict)
        writer_func : callable
            Function to write the data
        """
        if self.inherited_metadata.get(metadata_type, False) and self.session > 0:
            # For inherited files, only write if they don't exist or if overwrite is True
            full_path = self._get_metadata_filename(metadata_type) + '.' + metadata_type.split('.')[-1]
            if self.overwrite or not os.path.exists(full_path):
                writer_func(data)
        else:
            # For non-inherited files, always write
            writer_func(data)

    def write(self):
        """Save dataset in BIDS format"""
        super().write()
        
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        # Make BIDS compatible file names  
        name = self.datapath + self.subject_name + '_' + self.task + '_'
        if self.session > 0:
            name = name + 'ses-' + str(int(self.session)).zfill(self.n_digits) + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # Write non-inherited metadata files
        self.channels.to_csv(name + 'channels.tsv', sep='\t', index=False, header=True)
        with open(name + self.datatype + '.json', 'w') as f:
            json.dump(self.emg_sidecar, f)

        # Handle inherited files
        if self.session > 0:
            session_name = self._get_session_prefix()
            
            # Write electrodes.tsv if inherited
            if self.inherited_metadata.get('electrodes', False):
                if self.overwrite or not os.path.exists(session_name + '_electrodes.tsv'):
                    self.electrodes.to_csv(session_name + '_electrodes.tsv', sep='\t', index=False, header=True)
            else:
                self.electrodes.to_csv(name + 'electrodes.tsv', sep='\t', index=False, header=True)
            
            # Write coordsystem.json if inherited
            if self.inherited_metadata.get('coordsystem', False):
                if self.overwrite or not os.path.exists(session_name + '_coordsystem.json'):
                    with open(session_name + '_coordsystem.json', 'w') as f:
                        json.dump(self.coord_sidecar, f)
            else:
                with open(name + 'coordsystem.json', 'w') as f:
                    json.dump(self.coord_sidecar, f)
        else:
            # For non-session data, write all files in datapath
            self.electrodes.to_csv(name + 'electrodes.tsv', sep='\t', index=False, header=True)
            with open(name + 'coordsystem.json', 'w') as f:
                json.dump(self.coord_sidecar, f)

        # Write edf file
        self.emg_data.write(name + self.datatype + '.edf')

    def read(self):
        """Import data from BIDS dataset"""
        super().read()

        # Make BIDS compatible file names   
        name = self.datapath + self.subject_name + '_' + self.task + '_'
        if self.session > 0:
            name = name + 'ses-' + str(int(self.session)).zfill(self.n_digits) + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # Read non-inherited metadata files
        if os.path.isfile(name + 'channels.tsv'):
            self.channels = pd.read_table(name + 'channels.tsv')
        if os.path.isfile(name + self.datatype + '.json'):
            with open(name + self.datatype + '.json', 'r') as f:
                self.emg_sidecar = json.load(f)

        # Handle inherited files
        if self.session > 0:
            session_name = self._get_session_prefix()
            
            # Read electrodes.tsv
            if self.inherited_metadata.get('electrodes', False):
                if os.path.isfile(session_name + '_electrodes.tsv'):
                    self.electrodes = pd.read_table(session_name + '_electrodes.tsv')
            elif os.path.isfile(name + 'electrodes.tsv'):
                self.electrodes = pd.read_table(name + 'electrodes.tsv')
            
            # Read coordsystem.json
            if self.inherited_metadata.get('coordsystem', False):
                if os.path.isfile(session_name + '_coordsystem.json'):
                    with open(session_name + '_coordsystem.json', 'r') as f:
                        self.coord_sidecar = json.load(f)
            elif os.path.isfile(name + 'coordsystem.json'):
                with open(name + 'coordsystem.json', 'r') as f:
                    self.coord_sidecar = json.load(f)
        else:
            # For non-session data, read from datapath
            if os.path.isfile(name + 'electrodes.tsv'):
                self.electrodes = pd.read_table(name + 'electrodes.tsv')
            if os.path.isfile(name + 'coordsystem.json'):
                with open(name + 'coordsystem.json', 'r') as f:
                    self.coord_sidecar = json.load(f)

        # Read edf file
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



class bids_neuromotion_recording(bids_emg_recording):
    """
    Class for handling neuromotion simulation data in BIDS format.
    Inherits from bids_emg_recording and adds support for additional simulation-specific files.
    """
    def __init__(self, 
                 subject=1, 
                 task='isometric', 
                 datatype='emg', 
                 session=-1, 
                 run=1,
                 data_obj=None,
                 root='./',
                 datasetname='my-data',
                 overwrite=False,
                 n_digits=2,
                 inherited_metadata=None):
        
        # If no inherited_metadata is provided, use all inheritable files
        if inherited_metadata is None:
            inherited_metadata = self.INHERITABLE_FILES
        
        super().__init__(
            subject=subject,
            task=task,
            datatype=datatype,
            session=session,
            run=run,
            data_obj=data_obj,
            root=root,
            datasetname=datasetname,
            overwrite=overwrite,
            n_digits=n_digits,
            inherited_metadata=inherited_metadata
        )
        
        # Process name and session input
        subject_name = 'sub' + '-' + 'sim' + str(subject).zfill(n_digits)
        if session < 0:
            datapath = self.root + '/' + subject_name + '/' + datatype + '/'
        else:
            ses_name = 'ses' + '-' + str(session).zfill(n_digits)
            datapath = self.root + '/' + subject_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        self.subject_name = subject_name

        # Initialize additional simulation-specific attributes
        self.spikes = pd.DataFrame(columns=['source_id', 'spike_time'])
        self.motor_units = pd.DataFrame(columns=['source_id', 'recruitment_threshold', 'depth', 'innervation_zone', 'fibre_density', 'fibre_length', 'conduction_velocity', 'angle'])
        self.internals = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.internals_sidecar = pd.DataFrame(columns=['name', 'type', 'units', 'description'])
        self.simulation_sidecar = {}

    def write(self):
        """Override write method to include simulated data"""
        # Call parent's write method to handle standard BIDS files
        super().write()
        
        # Make BIDS compatible file names
        name = self.datapath + self.subject_name + '_' + self.task + '_'
        if self.session > 0:
            name = name + 'ses-' + str(int(self.session)).zfill(self.n_digits) + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # Write simulation-specific files
        self.spikes.to_csv(name + 'spikes.tsv', sep='\t', index=False, header=True)
        self.motor_units.to_csv(name + 'motorunits.tsv', sep='\t', index=False, header=True)
        self.internals_sidecar.to_csv(name + 'internals.tsv', sep='\t', index=False, header=True)
        with open(name + 'simulation.json', 'w') as f:
            json.dump(self.simulation_sidecar, f)
        self.internals.write(name + 'internals.edf')

    def read(self):
        """Override read method to include simulated data"""
        # Call parent's read method first
        super().read()
        
        # Make BIDS compatible file names
        name = self.datapath + self.subject_name + '_' + self.task + '_'
        if self.session > 0:
            name = name + 'ses-' + str(int(self.session)).zfill(self.n_digits) + '_'
        if self.run > 0:
            name = name + 'run-' + str(int(self.run)).zfill(self.n_digits) + '_'

        # Read simulation-specific files
        if os.path.isfile(name + 'spikes.tsv'):
            self.spikes = pd.read_table(name + 'spikes.tsv')
        if os.path.isfile(name + 'motorunits.tsv'):
            self.motor_units = pd.read_table(name + 'motorunits.tsv')
        if os.path.isfile(name + 'internals.tsv'):
            self.internals_sidecar = pd.read_table(name + 'internals.tsv')
        if os.path.isfile(name + 'simulation.json'):
            with open(name + 'simulation.json', 'r') as f:
                self.simulation_sidecar = json.load(f)
        if os.path.isfile(name + 'internals.edf'):
            self.internals = read_edf(name + 'internals.edf')


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
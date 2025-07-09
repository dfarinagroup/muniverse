import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from edfio import Edf, EdfSignal, read_edf


class bids_dataset:

    def __init__(
        self, datasetname="dataset-name", root="./", n_digits=2, overwrite=False
    ):

        self.root = root + datasetname
        self.datasetname = datasetname
        self.dataset_sidecar = {
            "Name": datasetname,
            "BIDSversion": self._get_bids_version(),
        }
        self.subjects_sidecar = self._set_participant_sidecar()
        self.subjects_data = pd.DataFrame(
            columns=["name", "age", "sex", "hand", "weight", "height"]
        )
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
        name = self.root + "/" + "participants.tsv"
        if self.overwrite or not os.path.isfile(name):
            self.subjects_data.to_csv(name, sep="\t", index=False, header=True)
        elif not self.overwrite and os.path.isfile(name):
            from_file = pd.read_table(name)
            if not from_file.equals(self.subjects_data):
                self.subjects_data.to_csv(name, sep="\t", index=False, header=True)
        # write participant.json
        name = self.root + "/" + "participants.json"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w") as f:
                json.dump(self.subjects_sidecar, f)
        # write dataset.json
        name = self.root + "/" + "dataset_description.json"
        if self.overwrite or not os.path.isfile(name):
            with open(name, "w") as f:
                json.dump(self.dataset_sidecar, f)

        return ()

    def read(self):
        """
        Import data from BIDS dataset

        """

        # read participant.tsv
        name = self.root + "/" + "participants.tsv"
        if os.path.isfile(name):
            self.subjects_data = pd.read_table(name, on_bad_lines="warn")
        # read participant.json
        name = self.root + "/" + "participants.json"
        if os.path.isfile(name):
            with open(name, "r") as f:
                self.subjects_sidecar = json.load(f)
        # read dataset.json
        name = self.root + "/" + "dataset_description.json"
        if os.path.isfile(name):
            with open(name, "r") as f:
                self.dataset_sidecar = json.load(f)

        return ()

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
                current.update(source.to_dict(orient="records")[0])  # assumes one row
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

        if field_name == "subjects_data":
            current.drop_duplicates(subset="name", keep="last")
            current.sort_values("name")

        # Update field
        setattr(self, field_name, current)

    def list_all_files(self, extension):
        """
        Summarize all files with a given extension that are part of a BIDS folder

        Args:
            extension (str): File extension to be filtered (e.g. 'edf' for all *.edf files)

        Returns:
            df (DataFrame): Data frame listing all files in the given folder
        """

        root = Path(self.root)
        files = list(root.rglob(f"*{extension}"))
        filenames = [f.name for f in files]
        paths = [f.resolve() for f in files]
        columns = ["sub", "ses", "task", "run", "data_type", "file_format", "file_path"]
        df = pd.DataFrame(np.nan, index=range(len(filenames)), columns=columns)
        df = df.astype(
            {
                "sub": "string",
                "ses": "string",
                "task": "string",
                "run": "string",
                "data_type": "string",
                "file_format": "string",
                "file_path": "string",
            }
        )

        # TODO some checks for testing for valid file names / extensions

        for i in np.arange(len(filenames)):
            splitname = filenames[i].split("_")
            df.loc[i, "file_path"] = str(paths[i].parent)
            df.loc[i, "file_name"] = str(paths[i])
            if splitname[1].split("-")[0] == "ses":
                df.loc[i, "sub"] = splitname[0].split("-")[1]
                df.loc[i, "ses"] = splitname[1].split("-")[1]
                df.loc[i, "task"] = splitname[2].split("-")[1]
                df.loc[i, "run"] = splitname[3].split("-")[1]
                df.loc[i, "data_type"] = splitname[4].split(".")[0]
                df.loc[i, "file_format"] = splitname[4].split(".")[1]
            else:
                df.loc[i, "sub"] = splitname[0].split("-")[1]
                df.loc[i, "task"] = splitname[1].split("-")[1]
                df.loc[i, "run"] = splitname[2].split("-")[1]
                df.loc[i, "data_type"] = splitname[3].split(".")[0]
                df.loc[i, "file_format"] = splitname[3].split(".")[1]

        return df

    def _set_participant_sidecar(self):
        """
        Return a template for initalizing the participant sidecar file

        """

        metadata = {
            "name": {"Description": "Unique subject identifier"},
            "age": {
                "Description": "Age of the participant at time of testing",
                "Unit": "years",
            },
            "sex": {
                "Description": "Biological sex of the participant",
                "Levels": {"F": "female", "M": "male", "O": "other"},
            },
            "handedness": {
                "Description": "handedness of the participant as reported by the participant",
                "Levels": {"L": "left", "R": "right"},
            },
            "weight": {"Description": "Body weight of the participant", "Unit": "kg"},
            "height": {"Description": "Body height of the participant", "Unit": "m"},
        }

        return metadata
    
    def _id_to_label(self, id):
        """ 
        Convert an ID (int) into a label (string)

        """

        if type(id) is not int or id > 10**self.n_digits - 1:
            raise ValueError("invalid subject ID")
        else:
            label = str(id).zfill(self.n_digits)   

        return label

    def _get_bids_version(self):
        """
        Get the BIDS version of your dataset

        """

        bids_version = "extension proposal for electromyography - BEP042"

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
    INHERITABLE_FILES = ["channels" ,"electrodes", "coordsystem"]

    def __init__(
        self,
        subject_id=1,
        subject_desc="",
        task_label="isometric",
        datatype="emg",
        session=None,
        run=1,
        root="./",
        datasetname="my-data",
        overwrite=False,
        n_digits=2,
        inherited_metadata=None,
        dataset_config=None
    ):

        super().__init__(
            datasetname=datasetname,
            root=root,
            overwrite=overwrite,
        )

        if isinstance(dataset_config, bids_dataset):
            self.root = dataset_config.root
            self.datasetname = dataset_config.datasetname
            self.subjects_data = dataset_config.subjects_data

        # Check if the function arguments are valid
        if type(subject_id) is not int or subject_id > 10**n_digits - 1 or subject_id < 0:
            raise ValueError("invalid subject ID")

        if session is not None:
            if type(session) is not int or session > 10**n_digits - 1 or session < 1:
                raise ValueError("invalid session ID")

        if run is not None:
            if type(run) is not int or run > 10**n_digits - 1 or run < 1:
                raise ValueError("invalid run ID")

        if datatype not in ["emg"]:
            raise ValueError("datatype must be emg")

        # Process name and session input
        subject_name = f"sub-{subject_desc}{self._id_to_label(subject_id)}"
        if session is None:
            datapath = self.root + "/" + subject_name + "/" + datatype + "/"
        else:
            session_name = f"ses-{self._id_to_label(session)}"
            datapath = (
                self.root + "/" + subject_name + "/" + session_name + "/" + datatype + "/"
            )

        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        self.n_digits = n_digits
        #self.subject_id = subject_id
        self.subject_label = f"{subject_desc}{self._id_to_label(subject_id)}"
        self.task_label = task_label
        #self.task = "task-" + task_label
        self.session = session
        self.run = run
        self.datatype = datatype
        self.emg_data = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])

        # Initialize metadata
        self.channels = pd.DataFrame(columns=["name", "type", "unit"])
        self.electrodes = pd.DataFrame(
            columns=["name", "x", "y", "z", "coordinate_system"]
        )
        self.emg_sidecar = self._init_emg_sidecar()
        self.coord_sidecar = {"EMGCoordinateSystem": [], "EMGCoordinateUnits": []}

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
        - Only channels.tsv, electrodes.tsv and coordsystem.json can be inherited
        - Inherited files are stored at session level
        - Non-inherited files are stored with their respective recording files
        """
        # Validate input
        invalid_files = [f for f in metadata_files if f not in self.INHERITABLE_FILES]
        if invalid_files:
            raise ValueError(
                f"Invalid metadata files for inheritance: {invalid_files}. "
                f"Valid options are: {self.INHERITABLE_FILES}"
            )

        # Set inheritance flags
        self.inherited_metadata = {f: True for f in metadata_files}

    def _get_bids_filename(self, datatype, extension):
        """
        Get a BIDS compatible filename
        
        """

        fname = f"sub-{self.subject_label}_"
        folder = f"{self.root}/sub-{self.subject_label}/" 
        if self.session is not None:
            fname = fname + f"ses-{self._id_to_label(self.session)}_"
            folder = folder + f"ses-{self._id_to_label(self.session)}/"

        if not self.inherited_metadata.get(datatype, False):     
            fname = fname + f"task-{self.task_label}_"
            folder = folder + f"{self.datatype}/"
            if self.run > 0:
                fname = fname + f"run-{self._id_to_label(self.run)}_"

        fname = fname + f"{datatype}.{extension}" 

        name = folder + fname   

        return name

    def _init_emg_sidecar(self):
        """
        Initalize EMG sidecar metadata

        """

        metadata = {
            "EMGPlacementScheme": [],
            "EMGReference": [],
            "SamplingFrequency": [],
            "PowerLineFrequency": [],
            "SoftwareFilters": [],
            "TaskName": [],
        }

        return metadata

    def write(self):
        """
        Save dataset in BIDS format
        
        """
        super().write()

        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        # Write files
        filename = self._get_bids_filename("channels", "tsv")
        self.channels.to_csv(filename, sep="\t", index=False, header=True)
        filename = self._get_bids_filename("emg","json")
        with open(filename, "w") as f:
            json.dump(self.emg_sidecar, f)
        filename = self._get_bids_filename("electrodes","tsv")
        self.electrodes.to_csv(filename, sep="\t", index=False, header=True)
        filename = self._get_bids_filename("coordsystem","json")
        with open(filename, "w") as f:
            json.dump(self.coord_sidecar, f)
        filename = self._get_bids_filename("emg","edf")
        self.emg_data.write(filename)

    def read(self):
        """
        Import data from BIDS dataset
        
        """
        super().read()

        filename = self._get_bids_filename("channels", "tsv")
        if os.path.isfile(filename):
            self.channels = pd.read_table(filename, on_bad_lines="warn")
        filename = self._get_bids_filename("emg","json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.emg_sidecar = json.load(f)
        filename = self._get_bids_filename("electrodes","tsv")
        if os.path.isfile(filename):
            self.electrodes = pd.read_table(filename, on_bad_lines="warn")
        filename = self._get_bids_filename("coordsystem","json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.coord_sidecar = json.load(f)
        filename = self._get_bids_filename("emg", "edf")
        if os.path.isfile(filename):
            self.emg_data = read_edf(filename)

    def set_metadata(self, field_name, source):

        super().set_metadata(field_name, source)

        if field_name == "channels":
            # Drop duplicates
            self.channels = self.channels.drop_duplicates(subset="name", keep="last")
        elif field_name == "electrodes":
            # Drop duplicates
            self.electrodes = self.electrodes.drop_duplicates(
                subset=["name", "coordinate_system"], keep="last"
            )
            # If the z coordinate exist make sure the ordering is correct
            if "z" in self.electrodes.columns:
                col = self.electrodes.pop("z")
                self.electrodes.insert(3, "z", col)

    def set_data(self, field_name, mydata, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            field_name (str): name of the field to be updated
            mydata (np.ndarry): data matrix (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds
        seconds = np.ceil(mydata.shape[0] / fsamp)
        signal = np.zeros([int(seconds * fsamp), mydata.shape[1]])
        signal[0 : mydata.shape[0], :] = mydata

        # Initalize
        edf = Edf([EdfSignal(signal[:, 0], sampling_frequency=fsamp)])

        for i in np.arange(1, signal.shape[1]):
            new_signal = EdfSignal(signal[:, i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)

        # Set data
        setattr(self, field_name, edf)

        return ()

    def read_data_frame(self, df, idx):
        self.subject_label = df.loc[idx, "sub"]
        #self.subject_id = int(re.search(r"\d+", df.loc[idx, "sub"]).group())
        #self.task = "task-" + df.loc[idx, "task"]
        self.task_label = df.loc[idx, "task"]
        self.datatype = df.loc[idx, "data_type"]
        self.run = int(df.loc[idx, "run"])
        self.datapath = df.loc[idx, "file_path"] + "/"
        if pd.isna(df.loc[0, "ses"]):
            self.session = None
        else:
            self.session = int(
                df.loc[idx, "ses"]
            )  # TODO catch if session odes not exist
        pass


class bids_neuromotion_recording(bids_emg_recording):
    """
    Class for handling neuromotion simulation data in BIDS format.
    Inherits from bids_emg_recording and adds support for additional simulation-specific files.
    """

    def __init__(
        self,
        subject_id=1,
        subject_desc="sim",
        task_label="isometric",
        datatype="emg",
        session=None,
        run=1,
        dataset_config=None,
        root="./",
        datasetname="my-data",
        overwrite=False,
        n_digits=2,
        inherited_metadata=None,
    ):

        # If no inherited_metadata is provided, use all inheritable files
        if inherited_metadata is None:
            inherited_metadata = self.INHERITABLE_FILES

        super().__init__(
            subject_id=subject_id,
            subject_desc=subject_desc,
            task_label=task_label,
            datatype=datatype,
            session=session,
            run=run,
            dataset_config=dataset_config,
            root=root,
            datasetname=datasetname,
            overwrite=overwrite,
            n_digits=n_digits,
            inherited_metadata=inherited_metadata,
        )


        # Initialize additional simulation-specific attributes
        self.spikes = pd.DataFrame(columns=["source_id", "spike_time"])
        self.motor_units = pd.DataFrame(
            columns=[
                "source_id",
                "recruitment_threshold",
                "depth",
                "innervation_zone",
                "fibre_density",
                "fibre_length",
                "conduction_velocity",
                "angle",
            ]
        )
        self.internals = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.internals_sidecar = pd.DataFrame(
            columns=["name", "type", "units", "description"]
        )
        self.simulation_sidecar = {}

    def write(self):
        """Override write method to include simulated data"""
        # Call parent's write method to handle standard BIDS files
        super().write()

        # Write simulation-specific files
        filename = self._get_bids_filename("spikes","tsv")
        self.spikes.to_csv(filename, sep="\t", index=False, header=True)
        filename = self._get_bids_filename("motorunits","tsv")
        self.motor_units.to_csv(filename, sep="\t", index=False, header=True)
        filename = self._get_bids_filename("internals","tsv")
        self.internals_sidecar.to_csv(filename, sep="\t", index=False, header=True)
        filename = self._get_bids_filename("simulation","json")
        with open(filename, "w") as f:
            json.dump(self.simulation_sidecar, f)
        filename = self._get_bids_filename("internals","edf")    
        self.internals.write(filename)

    def read(self):
        """Override read method to include simulated data"""
        # Call parent's read method first
        super().read()

        # Read simulation-specific files
        filename = self._get_bids_filename("spikes","tsv")
        if os.path.isfile(filename):
            self.spikes = pd.read_table(filename, on_bad_lines="warn")
        filename = self._get_bids_filename("motorunits","tsv")    
        if os.path.isfile(filename):
            self.motor_units = pd.read_table(
                filename, on_bad_lines="warn"
            )
        filename = self._get_bids_filename("internals","tsv")    
        if os.path.isfile(filename):
            self.internals_sidecar = pd.read_table(
                filename, on_bad_lines="warn"
            )
        filename = self._get_bids_filename("simulation","json")    
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                self.simulation_sidecar = json.load(f)
        filename = self._get_bids_filename("internals","edf")         
        if os.path.isfile(filename):
            self.internals = read_edf(filename)


class bids_decomp_derivatives(bids_emg_recording):

    def __init__(
        self,
        pipelinename="pipeline-name",
        format="standalone",
        config=None,
        datasetname="dataset-name",
        datatype="emg",
        subject_id=1,
        subject_desc = "",
        task_label="isometric",
        run=1,
        session=-None,
        desc_label="decomposed",
        root="./",
        overwrite=False,
        n_digits=2,
    ):

        # Check if the function arguments are valid
        if type(subject_id) is not int or subject_id > 10**n_digits - 1 or subject_id < 0:
            raise ValueError("invalid subject ID")

        if session is not None:
            if type(session) is not int or session > 10**n_digits - 1:
                raise ValueError("invalid session ID")

        if run is not None:
            if type(run) is not int or run > 10**n_digits - 1:
                raise ValueError("invalid session ID")

        if datatype not in ["emg"]:
            raise ValueError("datatype must be emg")

        # Process name and session input
        subject_name = f"sub-{subject_desc}{self._id_to_label(subject_id)}"
        if session is None:
            datapath = f"{subject_name}/{datatype}/"
        else:
            session_name = f"ses-{self._id_to_label(session)}" 
            datapath = f"{subject_name}/{session_name}/{datatype}/"

        # Store essential information for BIDS compatible folder structure in a dictonary
        self.datapath = datapath
        #self.subject_id = subject_id
        self.subject_label = f"{subject_desc}{self._id_to_label(subject_id)}"
        self.task = "task-" + task_label
        self.session = session
        self.desc = "desc-" + desc_label
        self.overwrite = overwrite
        self.n_digits = n_digits
        self.run = run
        self.datatype = datatype

        if isinstance(config, bids_emg_recording):
            self.root = config.root
            self.datasetname = config.datasetname
            self.n_digits = config.n_digits
            self.subject_label = config.subject_label
            self.task_label = config.task_label
            self.run = config.run
            self.datatype = config.datatype
            self.session = config.session

        # Store essential information for BIDS compatible folder structure in a dictonary
        if format == "standalone":
            self.datasetname = f"{datasetname}-{pipelinename}"
            self.root = f"{root}{self.datasetname}/"

        else:
            self.datasetname = datasetname
            self.root = f"{root}{datasetname}/derivatives/{pipelinename}/"

        self.derivative_datapath = self.root + datapath
        self.pipelinename = pipelinename

        self.source = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.spikes = pd.DataFrame(columns=["unit_id", "spike_time", "timestamp"])
        self.pipeline_sidecar = {"PipelineName": pipelinename}
        self.dataset_sidecar = {
            "Name": datasetname + "_" + pipelinename,
            "BIDSversion": self._get_bids_version(),
            "GeneratedBy": pipelinename,
        }

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.derivative_datapath):
            os.makedirs(self.derivative_datapath)

        name = self.derivative_datapath + self.subject_name + "_"
        if self.session is not None:
            name = name + f"ses-{self._id_to_label(self.session)}_"
        name = name + self.task + "_"
        if self.run is not None:
            name = name + f"run-{self._id_to_label(self.run)}_"
        name = f"{name}_{self.desc}_"    

        # write *_predictedspikes.tsv
        self.spikes.to_csv(
            name + "events.tsv", sep="\t", index=False, header=True
        )
        # write *_pipeline.json
        fname = name + self.datatype + ".json"
        with open(fname, "w") as f:
            json.dump(self.pipeline_sidecar, f)
        # write *_predictedsources.edf file
        self.source.write(name + self.datatype + ".edf")
        # write dataset.json
        fname = self.root + "/dataset.json"
        if self.overwrite or not os.path.isfile(fname):
            with open(fname, "w") as f:
                json.dump(self.dataset_sidecar, f)

    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_predictedspikes.tsv
        name = self.derivative_datapath + self.subject_name + "_"
        if self.session > 0:
            name = name + f"ses-{self._id_to_label(self.session)}_"
        name = name + self.task + "_"
        if self.run > 0:
            name = name + f"run-{self._id_to_label(self.run)}_"
        name = f"{name}_{self.desc}_"         

        # read *_predictedspikes.tsv
        fname = name + "events.tsv"
        if os.path.isfile(fname):
            self.spikes = pd.read_table(fname, on_bad_lines="warn")
        # read *_pipeline.json
        fname = name + self.datatype + ".json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.pipeline_sidecar = json.load(f)
        # read *.edf file
        fname = name + self.datatype + ".edf"
        if os.path.isfile(fname):
            self.source = read_edf(fname)
        # read dataset.json
        fname = self.root + "/" + "dataset.json"
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                self.dataset_sidecar = json.load(f)

    def add_spikes(self, spikes):
        """
        Convert a dictionary of spike times to long-format TSV-style DataFrame.

        Parameters:
            spike_dict (dict): {source_id: list of spike times}

        """
        rows = []
        for unit_id, spike_times in spikes.items():
            for t in spike_times:
                rows.append({"source_id": unit_id, "spike_time": t})

        frames = [self.spikes, pd.DataFrame(rows)]
        self.spikes = pd.concat(frames, ignore_index=True)
        self.spikes = self.spikes.drop_duplicates(subset=["source_id", "spike_time"])


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
        np_data[:, i] = edf_data.signals[idx[i]].data

    return np_data

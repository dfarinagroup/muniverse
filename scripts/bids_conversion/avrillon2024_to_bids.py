import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py, os, json
from muniverse import __license__, __version__
from muniverse.utils.bids_routines import *
from pathlib import Path

# ------------------------------------------ #
# ---------  Helper functions -------------- #
# ------------------------------------------ #
  
def get_grid_coordinates(grid_name):
    """
    Helper funcion to extract electrode coordinates given a grid type.
    
    """

    if grid_name == 'GR04MM1305':
        x = np.zeros(64)
        y = np.zeros(64)
        y[0:12]  = 0
        x[0:12]  = np.linspace(11*4,0,12)
        y[12:25] = 4
        x[12:25] = np.linspace(0,12*4,13)
        y[25:38] = 8
        x[25:38] = np.linspace(12*4,0,13)
        y[38:51] = 12
        x[38:51] = np.linspace(0,12*4,13)
        y[51:64] = 16
        x[51:64] = np.linspace(12*4,0,13)

    elif grid_name == "GR08MM1305":
        x = np.zeros(64)
        y = np.zeros(64)
        y[0:12]  = 0
        x[0:12]  = np.linspace(11*8,0,12)
        y[12:25] = 8
        x[12:25] = np.linspace(0,12*8,13)
        y[25:38] = 16
        x[25:38] = np.linspace(12*8,0,13)
        y[38:51] = 24
        x[38:51] = np.linspace(0,12*8,13)
        y[51:64] = 32
        x[51:64] = np.linspace(12*8,0,13)
           
    else:
        raise ValueError('The given grid_name has no reference')

    return(x,y)

def make_channel_metadata(
        fsamp = 2048, 
        muscle = 'right tibialis anterior', 
        IED = 4, 
        ngrid = 4
    ):
    """
    Helper function to manually curate chanel metadata.
    
    """

    # Define the columns of the *_channels.tsv file
    columns = ['name', 'type', 'units', 'description', 'sampling_frequency', 
                'signal_electrode', 'reference', 'group', 'target_muscle',
                'low_cutoff', 'high_cutoff', 'status']
    
    # Init dataframe containing the channels metadata
    df = pd.DataFrame(np.nan, index=range(64*ngrid + 2), columns=columns)
    df = df.astype({
        "name": "string", 
        "type": "string", 
        "units": "string",
        "description": "string", 
        "sampling_frequency": "float",
        "signal_electrode": "string", 
        "reference": "string",
        "group": "string", 
        "target_muscle": "string", 
        "low_cutoff": "float", 
        "high_cutoff": "float", 
        "status": "string"
    })
    # Set channel names
    df.loc[:df.shape[0], 'name'] = ['Ch' + str(i+1).zfill(3) for i in range(64*ngrid + 2)]
    # All the EMG channels
    df.loc[:ngrid*64-1, 'type'] = 'EMG'
    df.loc[:ngrid*64-1, 'units'] = 'mV'
    df.loc[:ngrid*64-1, 'description'] = 'monopolar EMG'
    df.loc[:df.shape[0], 'sampling_frequency'] = fsamp
    df.loc[:ngrid*64-1, 'signal_electrode'] = ['E' + str(i+1).zfill(3) for i in range(64*ngrid)]
    df.loc[:ngrid*64-1, 'reference'] = 'R1'
    df.loc[0:64*1-1, 'group'] = 'grid1'
    df.loc[64:64*2-1, 'group'] = 'grid2'
    df.loc[64*2:64*3-1, 'group'] = 'grid3'
    df.loc[64*3:64*4-1, 'group'] = 'grid4'
    df.loc[:ngrid*64-1, 'target_muscle'] = muscle
    df.loc[:ngrid*64-1, 'low_cutoff'] = 500
    df.loc[:ngrid*64-1, 'high_cutoff'] = 20
    df.loc[:df.shape[0], 'status'] = 'n/a'
    # Set the task channels
    df.loc[ngrid*64+0, 'description'] = 'Requested Path'
    df.loc[ngrid*64+1, 'description'] = 'Performed Path'
    df.loc[ngrid*64+0, 'units'] = 'percent MVC'
    df.loc[ngrid*64+1, 'units'] = 'percent MVC'
    df.loc[ngrid*64+0, 'type'] = 'MISC'
    df.loc[ngrid*64+1, 'type'] = 'MISC'
    
    return df

def make_electrode_metadata(
        ngrids, 
        gridname='GR04MM1305'
    ):
    """
    Helper function to curate the electrode metadata
    
    """

    # Define the columns of the electrode.tsv file
    columns = ["name", "x", "y", "coordinate_system"]

    # Init dataframe containing the electrode metadata
    df = pd.DataFrame(np.nan, index=range(64*ngrids + 2), columns=columns)
    df = df.astype({
        "name": "string", 
        "x": "float", 
        "y": "float",
        "coordinate_system": "string", 
    })


    # Loop over each electrode (of the four grids) and set metadata
    elecorode_idx = 0
    for i in np.arange(ngrids):
        (xg, yg) = get_grid_coordinates(gridname)
        for j in np.arange(64):
            df.loc[elecorode_idx, "name"] = f"E{str(elecorode_idx+1).zfill(3)}" # f'E' + str(elecorode_idx))
            # Map all electrode coordinates into the grid1 coordinate system
            df.loc[elecorode_idx, "coordinate_system"] = "grid1"
            if i==0: # Lateral-Proximal
                df.loc[elecorode_idx, "x"] = xg[j]
                df.loc[elecorode_idx, "y"] = yg[j]
            elif i==3: # Medial-Proximal
                y_shift = 20 if gridname == "GR04MM1305" else 40
                df.loc[elecorode_idx, "x"] = xg[j]
                df.loc[elecorode_idx, "y"] = yg[j] + y_shift
            elif i==1: # Lateral-Distal
                x_shift = 100 if gridname == "GR04MM1305" else 200
                y_shift = 36 if gridname == "GR04MM1305" else 72
                df.loc[elecorode_idx, "x"] = x_shift - xg[j]
                df.loc[elecorode_idx, "y"] = y_shift - yg[j]
            elif i==2: # Medial-Distal
                x_shift = 100 if gridname == "GR04MM1305" else 200
                y_shift = 16 if gridname == "GR04MM1305" else 32
                df.loc[elecorode_idx, "x"] = x_shift - xg[j]
                df.loc[elecorode_idx, "y"] = y_shift - yg[j]
            # Take care of the electrode index    
            elecorode_idx += 1    
   
   # Add the reference electrodes
    df.loc[ngrids*64+0, "name"] = "R1"
    df.loc[ngrids*64+1, "name"] = "R2"

    df.loc[ngrids*64+0, "coordinate_system"] = "lowerLeg"
    df.loc[ngrids*64+1, "coordinate_system"] = "lowerLeg"

    df.loc[ngrids*64+0, "x"] = 90
    df.loc[ngrids*64+1, "x"] = 95

    df.loc[ngrids*64+0, "y"] = 0
    df.loc[ngrids*64+1, "y"] = 0

    return df

def get_events_tsv(requested_path, fsamp, mvc_level, mvc_rate):
    """
    Helper function to convert the requested path
    into a events.tsv file
    
    """

    requested_path = requested_path.squeeze()

    columns = ["onset", "duration", "sample", "mvc_rate", "mvc_level", "event_type", "description"]
    df = pd.DataFrame(columns=columns)
    df = df.astype({
        "onset": "float", 
        "duration": "float", 
        "sample": "int",
        "mvc_rate": "float", 
        "mvc_level": "float",
        "event_type": "string", 
        "description": "string"
    })

    delta = 0.5
    path_0 = requested_path[0]
    path_max = np.max(requested_path)

    l_ramp = mvc_level / mvc_rate

    if mvc_level >= 70:
        l_plateau = 10
    elif mvc_level >= 50:
        l_plateau = 15
    else:
        l_plateau = 20

    idx_1 = np.argwhere(requested_path>path_0+delta).squeeze()[0]
    idx_2 = np.argwhere(requested_path>path_max-delta).squeeze()[0]
    idx_3 = np.argwhere(requested_path>path_max-delta).squeeze()[-1]
    idx_4 = np.argwhere(requested_path>path_0+delta).squeeze()[-1]

    mask1 = np.arange(idx_1,idx_2)
    m1, b1 = np.polyfit(mask1, requested_path[mask1], 1)
    mask2 = np.arange(idx_3,idx_4)
    m2, b2 = np.polyfit(mask2, requested_path[mask2], 1)

    idx_1 = int((0 - b1) / m1)
    idx_2 = int((mvc_level - b1) / m1)
    idx_3 = int((mvc_level - b2) / m2)
    idx_4 = int((0 - b2) / m2)

    df.loc[len(df)] = [
        np.round(idx_1/fsamp,6), 0, 
        idx_1, np.nan, np.nan, 
        "muscle_on",
        "Time at which the muscle is activated."
    ]
    df.loc[len(df)] = [
        np.round(idx_1/fsamp,6), l_ramp, 
        idx_1, mvc_rate, 0, 
        "linear_isometric_ramp",
        f"Linear ramp (rate: {mvc_rate} % MVC per s; duration: {l_ramp} s) of the isometric torque starting at 0 % MVC."
    ]
    df.loc[len(df)] = [
        np.round(idx_2/fsamp,6), l_plateau, 
        idx_2, 0, mvc_level, 
        "steady_isometric",
        f"Steady isometric torque at {mvc_level}% MVC for {l_plateau} s"
    ]
    df.loc[len(df)] = [
        np.round(idx_3/fsamp,6), l_ramp, 
        idx_3, -mvc_rate, mvc_level, 
        "linear_isometric_ramp",
        f"Linear ramp (rate: {-mvc_rate} % MVC per s; duration: {l_ramp} s) of the isometric torque starting at {mvc_level} % MVC."
    ]
    df.loc[len(df)] = [
        np.round(idx_4/fsamp,6), 0, 
        idx_4, np.nan, np.nan, 
        "muscle_off",
        "Time at which the muscle is deactivated."
    ]

    return df    

# ------------------------------------------ #
# --------  Dataset-level metadata --------- #
# ------------------------------------------ #

# Import the manually curated metadata
metadatapath = str(Path(__file__).parent.parent) + '/bids_metadata/' 
with open(metadatapath + 'avrillon_et_al_2024.json', 'r') as f:
    manual_metadata = json.load(f)

# Path to the original data (to be bidsified)
sourcepath = str(Path.home()) + '/Downloads/avrillon2024/'

# Sampling rate
fsamp = 2048
ngrids = 4

# List of subjects 
sub_id = [1, 2, 3, 4, 5, 6, 7, 8, 
          11, 12, 13, 14, 15, 16, 17, 18]
#sub_id = [4]
# Number of subjects
n_sub = len(sub_id)

# Set population level metadata
subjects_data = {
    'participant_id': [
        'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
        'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18'
    ], 
    'sex': [
        'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a',
        'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'
    ]
}
 
# Prepare the dataset-level README.md file
readme = """
# Avrillon et al 2024: HDsEMG recordings

BIDS-formatted version of the HDsEMG dataset published in *[Avrillon et al. 2024](https://doi.org/10.7554/eLife.97085.3)*. 
Two experimental sessions consisted of either a series of submaximal (10-80 percent MVC) 
isometric ankle dorsiflexions or isometric knee extensions. EMG signals were recorded from 
either the tibialis anterior (TA) or the vastus lateralis (VL) muscles using four arrays 
of 64 surface electrodes for a total of 256 electrodes.

### Population
16 young individuals volunteered to participate either in the experiment on the 
tibialis anterior (n=8; age: 27 +/- 3) or on the vastus lateralis (n=8; age: 27 +/- 10).

### Electrode placement
Surface EMG signals were recorded from the TA or the VL using 4 two-dimensional arrays of 
64 electrodes (GR04MM1305 for the TA; GR08MM1305 for the VL, 13×5 gold-coated electrodes with 
one electrode absent on a corner; interelectrode distance: 4 and 8 mm, respectively; OT Bioelettronica, Italy). 
The grids were positioned over the muscle bellies to cover the largest surface while staying away from 
the boundaries of the muscle identified by manual palpation. Before placing the electrodes, the 
skin was shaved and cleaned with an abrasive pad and water. A biadhesive foam layer was used to 
hold each array of electrodes onto the skin, and conductive paste filled the cavities of the 
adhesive layers to make skin-electrode contact.

### Tibialis anterior: ankle dorsiflexions
For the session of ankle dorsiflexions, participants sat on a massage table with the 
hips flexed at 45 degree, 0 degree being the hip neutral position, and the knees fully extended. 
The foot of the dominant leg (right in all participants) was fixed onto the pedal of an 
ankle dynamometer (OT Bioelettronica, Turin, Italy) positioned at 30 degree in the plantarflexion 
direction, 0 degree being the foot perpendicular to the shank. The thigh and the foot were 
fixed with inextensible Velcro straps. Force signals were recorded with a load cell 
(CCT Transducer s.a.s, Turin, Italy) connected in-series to the pedal using the same 
acquisition system as for the EMG recordings (EMG-Quattrocento; OT Bioelettronica, Italy).

### Vastus lateralis: knee extensions
For the session of knee extensions, participants sat on an instrumented chair with the hips 
flexed at 85 degree, 0 degree being the hip neutral position, and the knees flexed at 85 degree, 
0 degree being the knees fully extended. The torso and the thighs were fixed to the chair with 
Velcro straps and the tibia were positioned against a rigid resistance connected to force sensors 
(Metitur, Jyvaskyla, Finland). The force signals were recorded using the same acquisition 
system as for the EMG recordings.

### Coordinate systems
All electrode coordinates (reported in mm) have been converted to a common reference 
frame corresponding to the first EMG-array (*space-grid1*). 
The positions of the reference and ground electrodes are reported in a seperate 
coordinate system (*space-lowerLeg*) reported in percent of the lower leg length (knee-to-ankle). 

### Missing data
Contraction intensities 50, 60 and 70 % MVC are missing for subject 15.

### Conversion
The dataset has been converted semi-automatically using the [*MUniverse*](https://github.com/dfarinagroup/muniverse/tree/main) software.
See *dataset_description.json* for further details.

"""

# Prepare a events sidecar file
events_sidecar = {
    "onset": {
        "Description": "Onset time of the event in seconds from recording start.", 
        "Unit": "s"
    }, 
    "duration": {
        "Description": "Duration of the event in seconds. A value of zero means that the event is a dirac pulse", 
        "Unit": "s"
    }, 
    "sample": {
        "Description": "Sample index of the event onset (zero-indexing)", 
        "Unit": "samples"
    },
    "mvc_rate": {
        "Description": "Rate at which the torque changes in percent MVC per second",
        "Unit": "% MVC / s"
    }, 
    "mvc_level": {
        "Description": "MVC (maximum voluntary contraction) level at the onset of the event",
        "Unit": "% MVC"
    },
    "event_type": {
        "Description": "Event label.",
        "Levels": {
            "muscle_on": "The muscle is activated.",
            "muscle_off": "The muscle is deactivated.",
            "linear_isometric_ramp": "The isometric torque changes linearly over time with a fixed rate.",
            "steady_isometric": "Steady isometric contraction at a fixed MVC level."
        }
    },
    "description": {
        "Description": "Free text event description."
    }
}

# Dictonary of dataset-level metadata 
dataset_sidecar = manual_metadata["DatasetDescription"]
dataset_sidecar["GeneratedBy"][0]["Version"] = __version__
dataset_sidecar["GeneratedBy"][0]["License"] = __license__

# Handle the dataset level metadata
Avrillon_2024 = bids_dataset(
    datasetname='Avrillon_et_al_2024', 
    root=str(Path.home()) + '/Downloads/',
    overwrite=True
)
Avrillon_2024.readme = readme
Avrillon_2024.set_metadata(field_name='subjects_data', source=subjects_data)
Avrillon_2024.set_metadata(field_name='dataset_sidecar', source=dataset_sidecar)
Avrillon_2024.write()

# ------------------------------------------ #
# -------  Loop over all recordings -------- #
# ------------------------------------------ #
for i in np.arange(len(sub_id)):

    print(f"Bidsifying data of sub-{str(sub_id[i]).zfill(2)}")

    # List of files per subject
    filelist = [f for f in os.listdir(sourcepath) if f.startswith('S' + str(sub_id[i]) + '_')]

    # Loop over all recordings 
    for j in np.arange(len(filelist)):

        # Extract the task
        mvc_level = int(filelist[j].split('_')[1])
        task_label = f"isometric{mvc_level}percentmvc"
        print(f"    - Recording {j}: task-{task_label}")

        # Investigated muscle and selected electrode
        if sub_id[i] < 10:
            muscle = 'right tibialis anterior'
            grid   = 'GR04MM1305'
            ied    = 4
        else:
            muscle = 'right vastus lateralis'
            grid   = 'GR08MM1305'
            ied    = 8

        # Get the duration of the isometric plateau
        if mvc_level >= 70:
            l_plateau = 10
        elif mvc_level >= 50:
            l_plateau = 15
        else:
            l_plateau = 20   
      
        # Load data
        filename = sourcepath + filelist[j]
        try: 
            matfile = loadmat(filename, struct_as_record=False, squeeze_me=True)
            data = matfile['signal'].data
            target = matfile['signal'].target
            path = matfile['signal'].path
        except NotImplementedError:
            with h5py.File(filename, "r") as f:
                signal = f['signal']
                data = np.array(signal['data']).T
                target = np.array(signal['target']).T
                path = np.array(signal['path']).T

        emg_data = np.zeros((64*ngrids+2,data.shape[1]))
        emg_data[:64*ngrids,:] = data[:64*ngrids,:] / 150 # Divide by the gain to obtain EMG amplitudes in mV
        emg_data[64*ngrids,:] = target
        emg_data[64*ngrids+1,:] = path
        
        # channel metadata
        ch_metadata = make_channel_metadata(fsamp=fsamp, muscle=muscle, IED=ied, ngrid=ngrids)
        # electrode metadata
        el_metadata = make_electrode_metadata(ngrids=4, gridname=grid)
        # space metadata
        coordsystem_metadata = manual_metadata["CoordSystemSidecar"] 
        # emg sidecar metadata
        emg_sidecar = manual_metadata["EMGSidecar"] 
        #emg_sidecar["RecordingDuration"] = emg_data.shape[1]/fsamp
        emg_sidecar["InterelectrodeDistance"] = ied
        emg_sidecar["TaskName"] = task_label
        emg_sidecar["TaskDescription"] = f"Trapezoidal contraction: MVC level: {mvc_level} % MVC; MVC rate during ramps: 5 % MVC / s; plateau duration: {l_plateau} s."
        # events metadata
        events = get_events_tsv(target, fsamp, mvc_level, mvc_rate=5)

        # Make a recording and add data and metadata
        emg_recording = bids_emg_recording(
            parent_dataset=Avrillon_2024,
            subject_label=str(sub_id[i]).zfill(2), 
            task_label=task_label, 
            datatype="emg",
            inherited_metadata=["electrodes.tsv", "coordsystem.json", "events.json"],
            inherited_level=["subject", "subject", "dataset"],
            overwrite=False
        )
        
        emg_recording.set_metadata(field_name='channels', source=ch_metadata)
        emg_recording.set_metadata(field_name='electrodes', source=el_metadata) 
        emg_recording.set_metadata(field_name='emg_sidecar', source=emg_sidecar)
        emg_recording.set_metadata(field_name='coord_sidecar', source=coordsystem_metadata, overwrite=True)
        emg_recording.set_data(field_name='emg_data', mydata=emg_data,fsamp=fsamp)
        emg_recording.set_metadata(field_name="events_sidecar", source=events_sidecar)
        emg_recording.set_metadata(field_name="events", source=events)
        emg_recording.write()

# ------------------------------------------ #
# ---------  Validate outputs -------------- #
# ------------------------------------------ #

err, warn, _ = Avrillon_2024.validate(
    print_errors=True,
    print_warnings=True,
    ignored_codes=["TSV_COLUMN_TYPE_REDEFINED"],
    ignored_fields=["HEDVersion", "StimulusPresentation", "DeviceSerialNumber"],
    ignored_files=[]
)

# 
print("The BIDS conversion has completed")
print(f"Your BIDS dataset contains {len(err)} errors and {len(warn)} warnings")




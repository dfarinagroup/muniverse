import numpy as np
import pandas as pd
import json
#import os
#from edfio import *
from muniverse import __license__, __version__
from muniverse.utils.bids_routines import *
from muniverse.utils._otb_io import _open_otb, _format_otb_channel_metadata
from pathlib import Path

# ------------------------------------------ #
# ---------  Helper functions -------------- #
# ------------------------------------------ #

def get_grid_coordinates(grid_name):
    """
    Helper function to extract electrode coordinates given a grid type.
    
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
           
    else:
        raise ValueError('The given grid_name has no reference')

    return(x,y)

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
            elif i==1: # Medial-Proximal
                y_shift = 20 
                df.loc[elecorode_idx, "x"] = xg[j]
                df.loc[elecorode_idx, "y"] = yg[j] + y_shift
            elif i==2: # Lateral-Distal
                x_shift = 100 
                y_shift = 36 
                df.loc[elecorode_idx, "x"] = x_shift - xg[j]
                df.loc[elecorode_idx, "y"] = y_shift - yg[j]
            elif i==3: # Medial-Distal
                x_shift = 100 
                y_shift = 16 
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
    #t_4 = (path_0 - b2) / m2

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

# Path to the original data (to be bidsified)
sourcepath = str(Path.home()) + '/Downloads/Supplementary_data-1/RAW_HDEMG_SIGNALS/'

# Import the manually curated metadata
metadatapath = str(Path(__file__).parent.parent) + '/bids_metadata/' 
with open(metadatapath + 'caillet_et_al_2023.json', 'r') as f:
    manual_metadata = json.load(f)

# Sampling rate
fsamp = 2048
# Number of subjects
n_sub = 6
# Number of tasks
n_mvc = 2

subjects_data = {
    'participant_id': [
        'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06'
    ], 
    'sex': [
        'M', 'M', 'M', 'M', 'M', 'M'
    ]
}

# Prepare the dataset-level README.md file
readme = """
# Caillet et al 2023: HDsEMG recordings

BIDS-formatted version of the HDsEMG dataset published in *[Caillet et al. 2023](https://doi.org/10.1523/ENEURO.0064-23.2023)*. 

### Population
Six healthy male subjects (age: 26$\pm$4 years; height: 174$\pm$7 cm; weight: 66$\pm$15 kg).

### Protocol description
Each participant performed two trapezoidal contractions at 30 percent and 50 percent MVC, 
with 120 s of rest in between, consisting of linear ramps up and down performed at 
5 percent per second and a plateau maintained for 20 and 15 s at 30 percent and 
50 percent MVC, respectively. The order of the contractions was randomized.

### Electrode placement
First, the skin was shaved, abrased and cleansed with 70 percent ethyl alcohol.
Next, four grids (64 channels) were carefully positioned side-to-side with a 4-mm distance between the 
electrodes at the edges of adjacent grids. The 256 electrodes were centered on the 
muscle belly (right tibialis anterior) and laid within the muscle perimeter identified 
through palpation. Two bands damped with water were placed around the ankle as 
ground (R2) and reference (R1) electrodes. 

### Set-up description
The participant sat on a massage table with the hips flexed at 30 degrees, 0 degrees being 
the hip neutral position, and their knees fully extended. We fixed the foot of 
the dominant leg (right in all participants) onto the pedal of a commercial dynamometer (OT Bioelettronica) 
positioned at 30 degrees in the plantarflexion direction, 0 degrees being the foot 
perpendicular to the shank. The thigh was fixed to the massage table with an 
inextensible 3-cm-wide Velcro strap. The foot was fixed to the pedal with inextensible 
straps positioned around the proximal phalanx, metatarsal, and cuneiform. Force signals 
were recorded with a load cell (CCT Transducer s.a.s.) connected in-series to the pedal 
using the same acquisition system as for the HD-EMG recordings. The dynamometer was 
positioned according to the participant's lower limb length and secured to the massage table 
to avoid any motion during the contractions. 

### Missing data
There is no 50 % MVC ramp-and-hold contraction for the second subject.

### Coordinate systems
All electrode coordinates (reported in mm) have been converted to a common reference 
frame corresponding to the first EMG-array (*space-grid1*). 
The positions of the reference and ground electrodes are reported in a separate 
coordinate system (*space-lowerLeg*) reported as a percentage of the lower leg length. 

### Conversion
The dataset has been converted semi-automatically using the [*MUniverse*](https://github.com/dfarinagroup/muniverse/tree/main) software.
See *dataset_description.json* for further details.

"""

# Prepare an events sidecar file
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

dataset_sidecar = manual_metadata["DatasetDescription"] 

Caillet_2023 = bids_dataset(datasetname='Caillet_et_al_2023', root=str(Path.home()) + '/Downloads/')
Caillet_2023.set_metadata(field_name='subjects_data', source=subjects_data)
Caillet_2023.set_metadata(field_name='dataset_sidecar', source=dataset_sidecar)
Caillet_2023.readme = readme
Caillet_2023.write()

# ------------------------------------------ #
# -------  Loop over all recordings -------- #
# ------------------------------------------ #
for i in np.arange(n_sub):

    print(f"Bidsifying data of sub-{str(i+1).zfill(2)}")

    for j in np.arange(n_mvc):

        # There is no 50 percent MVC for the second subject
        if i==1 and j==1:
            continue

        folder = 'S' + str(i+1) + '/'

        if j==0:
            filename = 'S'  + str(i+1) + '_30MVC.otb+'
            task_label = 'isometric30percentmvc'
            mvc_level = 30
            plateau = 20
        elif j==1:
            filename = 'S'  + str(i+1) + '_50MVC.otb+'
            task_label = 'isometric50percentmvc'
            mvc_level = 50
            plateau = 15

        print(f"    - Recording {j}: task-{task_label}")    

        # Import data from otb+ file
        ngrids = 4
        fname =  sourcepath + folder + filename
        (data, metadata) = _open_otb(fname, ngrids)

        # channel metadata
        ch_metadata = _format_otb_channel_metadata(data,metadata,ngrids)
        ch_metadata.drop(columns="interelectrode_distance", axis=1, inplace=True)
        ch_metadata.loc[:255, "target_muscle"] = "right tibialis anterior"
        # electrode metadata
        el_metadata = make_electrode_metadata(ngrids=4)
        # space metadata
        coordsystem_metadata = manual_metadata["CoordSystemSidecar"] 
        # emg sidecar metadata
        emg_sidecar = manual_metadata["EMGSidecar"] 
        emg_sidecar['SamplingFrequency'] =  int(metadata['device_info']['SampleFrequency'])
        emg_sidecar['SoftwareVersions'] = metadata['subject_info']['software_version']
        emg_sidecar['ManufacturersModelName'] = metadata['device_info']['Name']
        emg_sidecar["TaskName"] = task_label
        emg_sidecar["TaskDescription"] = f"Trapezoidal contraction: MVC level: {mvc_level} % MVC; MVC rate during ramps: 5 % MVC / s; plateau duration: {plateau} s."
        #emg_sidecar["RecordingDuration"] = data.shape[1]/fsamp
        # events metadata
        indices = [i for i, s in enumerate(ch_metadata["description"]) if "requested path" in s]
        target = data[indices[0], :]
        events = get_events_tsv(target, fsamp, mvc_level, mvc_rate=5)

        # Make a recording and add data and metadata
        emg_recording = bids_emg_recording(
            parent_dataset=Caillet_2023,
            subject_label=str(i+1).zfill(2), 
            task_label=task_label, 
            datatype="emg",
            inherited_metadata=["electrodes.tsv", "coordsystem.json", "events.json"],
            inherited_level=["subject", "subject", "dataset"]
        )
        emg_recording.set_metadata(field_name='channels', source=ch_metadata)
        emg_recording.set_metadata(field_name='electrodes', source=el_metadata) 
        emg_recording.set_metadata(field_name='emg_sidecar', source=emg_sidecar)
        emg_recording.set_metadata(field_name='coord_sidecar', source=coordsystem_metadata, overwrite=True)
        emg_recording.set_data(field_name='emg_data', mydata=data,fsamp=2048)
        emg_recording.set_metadata(field_name="events_sidecar", source=events_sidecar)
        emg_recording.set_metadata(field_name="events", source=events)

        emg_recording.write()

# ------------------------------------------ #
# ---------  Validate outputs -------------- #
# ------------------------------------------ #
err, warn, _ = Caillet_2023.validate(
    print_errors=True,
    print_warnings=True,
    ignored_codes=["TSV_COLUMN_TYPE_REDEFINED"],
    ignored_fields=["HEDVersion", "StimulusPresentation", "DeviceSerialNumber"],
    ignored_files=[]
)

print("The BIDS conversion has completed")
print(f"Your BIDS dataset contains {len(err)} errors and {len(warn)} warnings")




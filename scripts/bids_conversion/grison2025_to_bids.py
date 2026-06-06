import numpy as np
import pandas as pd
import json
from scipy.signal import find_peaks
from muniverse import __license__, __version__
from muniverse.utils.bids_routines import *
from muniverse.utils._otb_io import _open_otb, _format_otb_channel_metadata
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
            
            if i==0: # Proximal
                df.loc[elecorode_idx, "coordinate_system"] = "grid1"
                df.loc[elecorode_idx, "x"] = xg[j]
                df.loc[elecorode_idx, "y"] = yg[j]
            elif i==1: # Distal (is rotated by 90 degree)
                df.loc[elecorode_idx, "coordinate_system"] = "grid2"
                x_shift = 48 
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

def get_events_tsv(requested_path, fsamp, mvc_level, duration_precision=1):
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

    peaks, _ = find_peaks(requested_path, height=0.9*mvc_level)

    start_idx = 0
    phase = 1

    for i in range(len(peaks)):

        if phase == len(peaks):
            end_idx = len(requested_path)
        else:
            end_idx = peaks[i] + int((peaks[i+1]-peaks[i])/2)

        mask = np.arange(start_idx, end_idx)
        path_i = requested_path[mask]

        idx_1 = np.argwhere(path_i>path_0+delta).squeeze()[0]
        idx_2 = np.argwhere(path_i>path_max-delta).squeeze()[0]
        idx_3 = np.argwhere(path_i>path_max-delta).squeeze()[-1]
        idx_4 = np.argwhere(path_i>path_0+delta).squeeze()[-1]

        mask1 = np.arange(idx_1,idx_2)
        m1, b1 = np.polyfit(mask1, path_i[mask1], 1)
        mask2 = np.arange(idx_3,idx_4)
        m2, b2 = np.polyfit(mask2, path_i[mask2], 1)

        idx_1 = int((0 - b1) / m1) + start_idx
        idx_2 = int((mvc_level - b1) / m1) + start_idx
        idx_3 = int((mvc_level - b2) / m2) + start_idx
        idx_4 = int((0 - b2) / m2) + start_idx
        #t_4 = (path_0 - b2) / m2

        df.loc[len(df)] = [
            np.round(idx_1/fsamp,6), 0, 
            idx_1, np.nan, np.nan, 
            "muscle_on",
            "Time at which the muscle is activated."
        ]
        l_ramp = np.round((idx_2-idx_1)/fsamp, duration_precision)
        mvc_rate = mvc_level/l_ramp
        df.loc[len(df)] = [
            np.round(idx_1/fsamp,6), l_ramp, 
            idx_1, mvc_rate, 0, 
            "linear_isometric_ramp",
            f"Linear ramp (rate: {mvc_rate} % MVC per s; duration: {l_ramp} s) of the isometric torque starting at 0 % MVC."
        ]
        l_plateau = np.round((idx_3-idx_2)/fsamp, duration_precision)
        df.loc[len(df)] = [
            np.round(idx_2/fsamp,6), l_plateau, 
            idx_2, 0, mvc_level, 
            "steady_isometric",
            f"Steady isometric torque at {mvc_level}% MVC for {l_plateau} s"
        ]
        l_ramp = np.round((idx_4-idx_3)/fsamp, duration_precision)
        mvc_rate = mvc_level/l_ramp
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

        start_idx = end_idx
        phase = phase + 1

    return df  

# ------------------------------------------ #
# --------  Dataset-level metadata --------- #
# ------------------------------------------ #

metadatapath = str(Path(__file__).parent.parent) + '/bids_metadata/' 

with open(metadatapath + 'grison_et_al_2025.json', 'r') as f:
    manual_metadata = json.load(f)

# Sampling rate
fsamp = 10240
# Number of subjects
n_sub = 1
# Number of trials
n_mvc = 1

mvc_levels = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70]

# Prepare the dataset-level README.md file
readme = """
# Grison et al 2025: HDsEMG recordings

BIDS-formatted version of a HDsEMG dataset corresponding to *[Grison et al. 2025](https://ieeexplore.ieee.org/abstract/document/10844937)*. 

### Overview
One healthy subjects performed 10 submaximal (10 to 70 percent MVC) isometric
ankle  dorsiflexions. EMG signals were recorded from the right tibialis anterior 
using two arrays of 64 surface electrodes (4 mm interelectrode distance, 13x5 configuration) 
for a total of 128 electrodes.

### Protocol description
The participant performed one, two, or three trapezoidal contractions (with repetitions being
specified by the run labels) at  10, 15, 20, 25, 30, 35, 40, 50, 60, and 70 percent MVC 
with 120 s of rest in between, consisting of linear ramps up and down performed at 
5 percent per second and a plateau maintained for 20 s up to 30 percent MVC, 15 s for 35 percent 
and 40 percent MVC, and 10 s from 50 percent to 70 percent MVC. The order of the 
contractions was randomized.

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

# Coordinate systems
All electrode coordinates (reported in mm) are reported in their respective 
grid coordinate system (*space-grid1* and *space-grid1*). 
Their relative positions as well as the positions of the reference
and ground electrodes are reported in a seperate  coordinate system (*space-lowerLeg*) 
reported in percent of the lower leg length. 

# Labeled motor unit spike trains
Labeld motor unit spike trains were derived from concurrently recorded invasive EMG
and curated by an experienced investigator (only available for *_run-01* of each trial).
The labeled motor unit spike trains are stored as a BIDS-derivative. 

# Conversion
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

sourcepath = str(Path.home()) + '/Downloads/S1/'

subjects_data = {'participant_id': ['sub-01'], 
            'sex': ['n/a']}
dataset_sidecar = manual_metadata["DatasetDescription"] #dataset_sidecar_template(ID='Caillet2023')
dataset_sidecar["GeneratedBy"][0]["Version"] = __version__
dataset_sidecar["GeneratedBy"][0]["License"] = __license__

Grison_2025 = bids_dataset(datasetname='Grison_et_al_2025', root=str(Path.home()) + '/Downloads/')
Grison_2025.set_metadata(field_name='subjects_data', source=subjects_data)
Grison_2025.set_metadata(field_name='dataset_sidecar', source=dataset_sidecar)
Grison_2025.readme = readme
Grison_2025.write()

# ------------------------------------------ #
# -------  Loop over all recordings -------- #
# ------------------------------------------ #

for i in np.arange(n_sub):

    print(f"Bidsifying data of sub-{str(i+1).zfill(2)}")

    for j in np.arange(len(mvc_levels)):

        mvc_level = mvc_levels[j]
        task_label = f"isometric{str(mvc_levels[j])}percentmvc"

        print(f"    - Recording {j}: task-{task_label}")

        # Import daata from otb+ file
        nadapter = 9
        fname =  sourcepath + str(mvc_levels[j]) + '/' + str(mvc_levels[j]) + 'mvc_semg.otb+'
        (data, metadata) = _open_otb(fname, nadapter)

        # Get and write channel metadata
        ch_metadata = _format_otb_channel_metadata(data,metadata,nadapter)
        ch_metadata.drop(columns="interelectrode_distance", axis=1, inplace=True)
        ch_metadata = ch_metadata[
            ch_metadata['target_muscle'].str.contains('Tibialis Anterior|n/a')
        ]
        idx = np.asarray(ch_metadata.index.to_list(),dtype=int)
        ch_metadata.loc[idx, "name"] = [f"Ch{str(i+1).zfill(3)}" for i in range(131)]
        ch_metadata.loc[idx[:128], "signal_electrode"] = [f"E{str(i+1).zfill(3)}" for i in range(128)]
        ch_metadata.loc[idx[:128], "target_muscle"] = "right tibialis anterior"
        ch_metadata.loc[idx[:64], "group"] = "grid1"
        ch_metadata.loc[idx[64:128], "group"] = "grid2"
        # Only keep the relevant channels
        data = data[idx, :]

        # Get electrode metadata
        el_metadata = make_electrode_metadata(ngrids=2)

        # Make the coordinate system sidecar file (here just a placeholder)
        coordsystem_metadata = manual_metadata["CoordSystemSidecar"] # {'EMGCoordinateSystem': 'local', 'EMGCoordinateUnits': 'mm'}

        # Make the emg sidecar file
        emg_sidecar = manual_metadata["EMGSidecar"] #emg_sidecar_template('Caillet2023')
        emg_sidecar['SamplingFrequency'] =  int(metadata['device_info']['SampleFrequency'])
        emg_sidecar['SoftwareVersions'] = metadata['subject_info']['software_version']
        emg_sidecar["TaskName"] = task_label
        emg_sidecar["TaskDescription"] = f"Trapezoidal contraction: MVC level: {mvc_level} % MVC; MVC rate during ramps: 10 % MVC / s."
        #emg_sidecar["RecordingDuration"] = data.shape[1]/fsamp

        # If there are repeated ramps, spilt up the signal into runs
        indices = [i for i, s in enumerate(ch_metadata["description"]) if "requested path" in s]
        target = data[indices[0], :]
        peaks, _ = find_peaks(target, height=0.9*mvc_level)
        
        # Loop over all runs
        start_idx = 0
        phase = 1
        for k in range(len(peaks)):

            print(f"       - Run: run-{str(k+1).zfill(2)}")

            if phase == len(peaks):
                end_idx = data.shape[1]
            else:
                end_idx = peaks[k] + int((peaks[k+1]-peaks[k])/2)

            data_k = data[:,start_idx:end_idx]
            target_k= target[start_idx:end_idx]    

            events = get_events_tsv(target_k, fsamp, mvc_level)

            # Make a recording and add data and metadata
            emg_recording = bids_emg_recording(
                parent_dataset=Grison_2025,
                subject_label=str(i+1).zfill(2), 
                task_label=task_label, 
                run_label=str(k+1).zfill(2),
                datatype="emg",
                inherited_metadata=["electrodes.tsv", "coordsystem.json", "events.json"],
                inherited_level=["subject", "subject", "dataset"]
            )
            emg_recording.set_metadata(field_name='channels', source=ch_metadata)
            emg_recording.set_metadata(field_name='electrodes', source=el_metadata) 
            emg_recording.set_metadata(field_name='emg_sidecar', source=emg_sidecar)
            emg_recording.set_metadata(field_name='coord_sidecar', source=coordsystem_metadata, overwrite=True)
            emg_recording.set_data(field_name='emg_data', mydata=data_k,fsamp=emg_sidecar['SamplingFrequency'])
            emg_recording.set_metadata(field_name="events_sidecar", source=events_sidecar)
            emg_recording.set_metadata(field_name="events", source=events)

            emg_recording.write()

            # Add the annotaed spike labels
            ref_labels = bids_decomp_derivatives(
                parent_recording=emg_recording,
                pipelinename="Manual",
                format="subdir",
                inherited_metadata=["events.json"],
                inherited_level=["subject"]
            )
            pipeline_desc = "Semi-automated expert-curated reference decomposition based on invasive EMG"
            ref_labels.dataset_sidecar["GeneratedBy"][0]["Description"] = pipeline_desc
            # Load data and reformat it to BIDS events format
            fname =  f"{sourcepath}{mvc_levels[j]}/{mvc_levels[j]}_spike_times.csv"
            label_df = pd.read_csv(fname)
            label_df = label_df.rename(
                columns={"source_id": "unit_id", "spike_time": "sample"}
            )
            label_df = label_df.loc[
                (label_df["sample"] > start_idx) & (label_df["sample"] < end_idx)
            ]
            label_df["sample"] = label_df["sample"] - start_idx
            if label_df.empty:
                pass
            else:
                ref_labels.add_spikes(label_df, fsamp=fsamp)
                ref_labels.write()

            phase = phase + 1
            start_idx = end_idx

# ------------------------------------------ #
# ---------  Validate outputs -------------- #
# ------------------------------------------ #

err, warn, _ = Grison_2025.validate(
    print_errors=True,
    print_warnings=True,
    ignored_codes=["TSV_COLUMN_TYPE_REDEFINED"],
    ignored_fields=["HEDVersion", "StimulusPresentation", "DeviceSerialNumber"],
    ignored_files=[]
)

print("The BIDS conversion has completed")
print(f"Your BIDS dataset contains {len(err)} errors and {len(warn)} warnings")




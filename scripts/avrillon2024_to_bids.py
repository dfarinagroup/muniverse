import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
import os
import json
from edfio import *
from src.data_preparation.data2bids import *
from src.data_preparation.otb_io import open_otb, format_otb_channel_metadata
#from .sidecar_templates import emg_sidecar_template, dataset_sidecar_template
from pathlib import Path

# Helper function for getting electrode coordinates
def get_grid_coordinates(grid_name):

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

# Helper function for making channel metadata
def make_channel_metadata(fsamp = 2048, muscle = 'Tibialis Anterior', IED = '4 mm', grid_name = 'GR04MM1305'):
    columns = ['name', 'type', 'unit', 'description', 'sampling_frequency', 
                'signal_electrode', 'reference_electrode', 'group', 'target_muscle',
                'interelectrode_distance', 'grid_name', 'low_cutoff', 'high_cutoff', 'status']
    
    ngrid = 4

    df = pd.DataFrame(np.nan, index=range(64*ngrid + 2), columns=columns)
    df = df.astype({"name": "string", "type": "string", "unit": "string",
                    "description": "string", "sampling_frequency": "float",
                    "signal_electrode": "string", "reference_electrode": "string",
                    "group": "string", "target_muscle": "string", "interelectrode_distance": "string",
                    "grid_name": "string", "low_cutoff": "float", "high_cutoff": "float", "status": "string"})

    df.loc[:df.shape[0], 'name'] = ['Ch' + str(i+1).zfill(2) for i in range(64*ngrid + 2)]
    df.loc[:ngrid*64-1, 'type'] = 'EMG'
    df.loc[:ngrid*64-1, 'unit'] = 'mV'
    df.loc[:ngrid*64-1, 'description'] = 'ElectroMyoGraphy'
    df.loc[:df.shape[0], 'sampling_frequency'] = fsamp
    df.loc[:ngrid*64-1, 'signal_electrode'] = ['E' + str(i+1).zfill(2) for i in range(64*ngrid)]
    df.loc[:ngrid*64-1, 'reference_electrode'] = 'R1'
    df.loc[0:64*1-1, 'group'] = 'Grid1'
    df.loc[64:64*2-1, 'group'] = 'Grid2'
    df.loc[64*2:64*3-1, 'group'] = 'Grid3'
    df.loc[64*3:64*4-1, 'group'] = 'Grid4'
    df.loc[:ngrid*64-1, 'target_muscle'] = muscle
    df.loc[:ngrid*64-1, 'interelectrode_distance'] = IED
    df.loc[:ngrid*64-1, 'grid_name'] = grid_name
    df.loc[:ngrid*64-1, 'low_cutoff'] = 500
    df.loc[:ngrid*64-1, 'high_cutoff'] = 20
    df.loc[:df.shape[0], 'status'] = 'n/a'
    #df.loc[ngrid*64+2, 'description'] = 'Auxilary force input'
    df.loc[ngrid*64+0, 'description'] = 'Requested Path'
    df.loc[ngrid*64+1, 'description'] = 'Performed Path'
    #df.loc[ngrid*64+2, 'unit'] = 'V'
    df.loc[ngrid*64+0, 'unit'] = 'percent MVC'
    df.loc[ngrid*64+1, 'unit'] = 'percent MVC'
    #df.loc[ngrid*64+2, 'type'] = 'MISC'
    df.loc[ngrid*64+0, 'type'] = 'MISC'
    df.loc[ngrid*64+1, 'type'] = 'MISC'
    
    return df

# Helper  function for making the electrode metadata
def make_electrode_metadata(ngrids):
    name              = []
    x                 = []
    y                 = []
    coordinate_system = []

    elecorode_idx = 0
    for i in np.arange(ngrids):
        (xg, yg) = get_grid_coordinates('GR04MM1305')
        for j in np.arange(64):
            elecorode_idx += 1
            name.append('E' + str(elecorode_idx))
            coordinate_system.append('Grid1')
            if i==0:
                x.append(xg[j])
                y.append(yg[j])
            elif i==1:
                x.append(xg[j])
                y.append(yg[j] + 20) 
            elif i==2:
                x.append(100 - xg[j])
                y.append(16 - yg[j]) 
            elif i==3:
                x.append(100 - xg[j])
                y.append(36 - yg[j])     
    name.append('R1')
    name.append('R2')
    x.append('n/a') 
    x.append('n/a') 
    y.append('n/a') 
    y.append('n/a') 
    coordinate_system.append('n/a') 
    coordinate_system.append('n/a')        
    el_metadata = {'name': name, 'x': x, 'y': y, 'coordinate_system': coordinate_system}

    return(el_metadata)

metadatapath = str(Path(__file__).parent.parent) + '/bids_metadata/' 

with open(metadatapath + 'avrillon_et_al_2024.json', 'r') as f:
    manual_metadata = json.load(f)


sourcepath = str(Path.home()) + '/Downloads/avrillon2024/'


# Subjects 
sub_id = [1, 2, 3, 4, 5, 6, 7, 8, 
          11, 12, 13, 14, 15, 16, 17, 18]

# Number of subjects
n_sub = 1 #len(sub_id)

#matfile = loadmat(sourcepath + 'S1_10_DF.otb+_decomp.mat_edited.mat', struct_as_record=False, squeeze_me=True)


subjects_data = {'name': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
                          'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18'], 
            'sex': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a',
                    'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']}
dataset_sidecar = manual_metadata["DatasetDescription"] #dataset_sidecar_template(ID='Caillet2023')

Avrillon_2024 = bids_dataset(datasetname='Avrillon_et_al_2024', root=str(Path.home()) + '/Downloads/')
Avrillon_2024.set_metadata(field_name='subjects_data', source=subjects_data)
Avrillon_2024.set_metadata(field_name='dataset_sidecar', source=dataset_sidecar)
Avrillon_2024.write()

for i in np.arange(len(sub_id)):

    filelist = [f for f in os.listdir(sourcepath) if f.startswith('S' + str(sub_id[i]) + '_')]
    print('sub-' + str(sub_id[i]))
    for j in np.arange(len(filelist)):

        if sub_id[i] < 10:
            muscle = 'Tibialis Anterior'
            grid   = 'GR04MM1305'
            ied    = '4 mm'
        else:
            muscle = 'Vastus Lateralis'
            grid   = 'GR08MM1305'
            ied    = '8 mm'

        task = 'isometric' + filelist[j].split('_')[1] + 'percentmvc'

        filename = sourcepath + filelist[j]
        try: 
            matfile = loadmat(filename, struct_as_record=False, squeeze_me=True)
            data = matfile['signal'].data
            target = matfile['signal'].target
            path = matfile['signal'].path
        except NotImplementedError:
            with h5py.File(filename, "r") as f:
                signal = f['signal']
                #data_ref = signal['data']
                data = np.array(signal['data']).T
                target = np.array(signal['target']).T
                path = np.array(signal['path']).T


        ngrids = 4

        emg_data = np.zeros((64*ngrids+2,data.shape[1]))
        #emg_data = data
        #emg_data = emg_data[:64*ngrids+2,:]
        emg_data[:64*ngrids,:] = data[:64*ngrids,:] / 150 # Manually add gain of the emg adapters
        #emg_data[64*ngrids,:] = emg_data[64*ngrids,:] / 0.5   # Manually add gain of the aux adapter    
        emg_data[64*ngrids,:] = target
        emg_data[64*ngrids+1,:] = path
        emg_data = emg_data.T

        # Import daata from otb+ file
        
        
        # Get and write channel metadata
        ch_metadata = make_channel_metadata(muscle=muscle, grid_name=grid, IED=ied)

        # Get electrode metadata
        el_metadata = make_electrode_metadata(ngrids=4)

        # Make the coordinate system sidecar file (here just a placeholder)
        coordsystem_metadata = manual_metadata["CoordSystemSidecar"] # {'EMGCoordinateSystem': 'local', 'EMGCoordinateUnits': 'mm'}

        # Make the emg sidecar file
        emg_sidecar = manual_metadata["EMGSidecar"] #emg_sidecar_template('Caillet2023')
        #emg_sidecar['SamplingFrequency'] =  int(metadata['device_info']['SampleFrequency'])
        #emg_sidecar['SoftwareVersions'] = metadata['subject_info']['software_version']
        #emg_sidecar['ManufacturerModelName'] = metadata['device_info']['Name']


        # Make a recording and add data and metadata
        emg_recording = bids_emg_recording(data_obj=Avrillon_2024,subject=sub_id[i], task=task, datatype='emg')
        emg_recording.set_metadata(field_name='channels', source=ch_metadata)
        emg_recording.set_metadata(field_name='electrodes', source=el_metadata) 
        emg_recording.set_metadata(field_name='emg_sidecar', source=emg_sidecar)
        emg_recording.set_metadata(field_name='coord_sidecar', source=coordsystem_metadata)
        emg_recording.set_data(field_name='emg_data', mydata=emg_data,fsamp=2048)

        emg_recording.write()

print('done')




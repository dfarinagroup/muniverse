import numpy as np
import pandas as pd
import os
from edfio import *
from data2bids import *
from otb_io import open_otb, format_otb_channel_metadata, format_subject_metadata
from sidecar_templates import emg_sidecar_template, dataset_sidecar_template
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

# Helper  function for making the electrode metadata
def make_electrode_metadata(ngrids):
    name              = []
    x                 = []
    y                 = []
    coordinate_system = []
    for i in np.arange(ngrids):
        (xg, yg) = get_grid_coordinates('GR04MM1305')
        for j in np.arange(64):
            name.append('E' + str(j+1))
            x.append(xg[j])
            y.append(yg[j])
            coordinate_system.append('Grid' + str(i+1))
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


# Define path and name of the BIDS structure
#bids_path = make_bids_path(subject=1, task='isometric-30-percent-mvc', datatype='emg', root='./data')

# Number of subjects
n_sub = 6
# Number of trials
n_mvc = 2

datapath = str(Path.home()) + '/Downloads/Supplementary_data-1/RAW_HDEMG_SIGNALS/'

subjects_data = {'name': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06'], 
            'sex': ['M', 'M', 'M', 'M', 'M', 'M']}
dataset_sidecar = dataset_sidecar_template(ID='Caillet2023')

Caillet_2023 = bids_dataset(datasetname='Caillet_et_al_2023', root='./')
Caillet_2023.set_metadata(field_name='subjects_data', source=subjects_data)
Caillet_2023.set_metadata(field_name='dataset_sidecar', source=dataset_sidecar)
Caillet_2023.write()

for i in np.arange(n_sub):
    for j in np.arange(n_mvc):

        # There is no 50 percent MVC for the second subject
        if i==1 and j==1:
            continue

        folder = 'S' + str(i+1) + '/'

        if j==0:
            filename = 'S'  + str(i+1) + '_30MVC.otb+'
            task = 'isometric30percentmvc'
        elif j==1:
            filename = 'S'  + str(i+1) + '_50MVC.otb+'
            task = 'isometric50percentmvc'


        # Import daata from otb+ file
        ngrids = 4
        fname =  datapath + folder + filename
        (data, metadata) = open_otb(fname, ngrids)

        # Get and write channel metadata
        ch_metadata = format_otb_channel_metadata(data,metadata,ngrids)

        # Get electrode metadata
        el_metadata = make_electrode_metadata(ngrids=4)

        # Make the coordinate system sidecar file (here just a placeholder)
        coordsystem_metadata = {'EMGCoordinateSystem': 'local', 'EMGCoordinateUnits': 'mm'}

        # Make the emg sidecar file
        emg_sidecar = emg_sidecar_template('Caillet2023')
        emg_sidecar['SamplingFrequency'] =  int(metadata['device_info']['SampleFrequency'])
        emg_sidecar['SoftwareVersions'] = metadata['subject_info']['software_version']
        emg_sidecar['ManufacturerModelName'] = metadata['device_info']['Name']


        # Make a recording and add data and metadata
        emg_recoring = bids_emg_recording(data_obj=Caillet_2023,subject=int(i+1), task=task, datatype='emg')
        emg_recoring.set_metadata(field_name='channels', source=ch_metadata)
        emg_recoring.set_metadata(field_name='electrodes', source=el_metadata) 
        emg_recoring.set_metadata(field_name='emg_sidecar', source=emg_sidecar)
        emg_recoring.set_metadata(field_name='coord_sidecar', source=coordsystem_metadata)
        emg_recoring.set_data(field_name='emg_data', mydata=data,fsamp=2048)

        emg_recoring.write()

print('done')




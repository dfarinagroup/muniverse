from data2bids import *
from sidecar_templates import emg_sidecar_template, dataset_sidecar_template
from edfio import *
import numpy as np

# Define path and name of the BIDS structure
bids_path = make_bids_path(subject=1, task='isometric-30-percent-mvc', datatype='emg', root='./data')

# Import daata from otb+ file
ngrids = 4
(data, metadata) = open_otb('./../utils/MVC_30MVC.otb+',ngrids)

# Get and write channel metadata
ch_metadata = format_otb_channel_metadata(data,metadata,ngrids)
make_channel_tsv(bids_path,ch_metadata)

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

# Generate and write electrode metadata
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
make_electrode_tsv(bids_path, el_metadata)        

# Make the coordinate system sidecar file (here just a placeholder)
coordsystem_metadata = {'EMGCoordinateSystem': 'local'}
make_coordinate_system_json(bids_path, coordsystem_metadata)

# Make the emg sidecar file
emg_sidecar = emg_sidecar_template('Caillet2023')
make_emg_json(bids_path,emg_sidecar)

# Make subject sidecar file 
make_participant_json(bids_path,'exp')

# Save individual subject file
subject = {}
subject['name'] = bids_path['subject']
subject['age']  = 'n/a'
subject['sex'] = 'M'
subject['hand'] = 'n/a'
subject['weight'] = 'n/a'
subject['height'] = 'n/a'
make_participant_tsv(bids_path, subject)

# Make dataset sidecar file
dataset_metadata = dataset_sidecar_template('n/a')
make_dataset_description_json(bids_path, dataset_metadata)

# Convert the raw data to an .edf file
write_edf(data = data, fsamp = 2048, ch_names = ch_metadata['name'], bids_path = bids_path)






import numpy as np
import pandas as pd
from edfio import *
from data2bids import emg_bids_io
from otb_io import open_otb, format_otb_channel_metadata, format_subject_metadata
from sidecar_templates import emg_sidecar_template, dataset_sidecar_template


# Define path and name of the BIDS structure
#bids_path = make_bids_path(subject=1, task='isometric-30-percent-mvc', datatype='emg', root='./data')
bids_gen = emg_bids_io(subject=1, task='isometric-30-percent-mvc', datatype='emg', root='./data')
bids_gen.read()
# Import daata from otb+ file
ngrids = 4
(data, metadata) = open_otb('./../utils/MVC_30MVC.otb+',ngrids)

# Get and write channel metadata
ch_metadata = format_otb_channel_metadata(data,metadata,ngrids)
bids_gen.add_channel_metadata(ch_metadata)
bids_gen.set_raw_data(data,2048)
bids_gen.write()

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
bids_gen.make_electrode_tsv(el_metadata)        

# Make the coordinate system sidecar file (here just a placeholder)
coordsystem_metadata = {'EMGCoordinateSystem': 'local', 'EMGCoordinateUnits': 'mm'}
bids_gen.make_coordinate_system_json(coordsystem_metadata)

# Make the emg sidecar file
emg_sidecar = emg_sidecar_template('Caillet2023')
emg_sidecar['SamplingFrequency'] =  int(metadata['device_info']['SampleFrequency'])
emg_sidecar['SoftwareVersions'] = metadata['subject_info']['software_version']
emg_sidecar['ManufacturerModelName'] = metadata['device_info']['Name']
bids_gen.make_emg_json(emg_sidecar)

# Make subject sidecar file 
bids_gen.make_participant_json('exp')

# Save individual subject file
subject = format_subject_metadata(bids_gen.subject, metadata)
bids_gen.make_participant_tsv(subject)

# Make dataset sidecar file
dataset_metadata = dataset_sidecar_template('n/a')
bids_gen.make_dataset_description_json(dataset_metadata)

# Convert the raw data to an .edf file
bids_gen.emg_to_edf(data = data[:,:ngrids*64], 
                    fsamp = int(metadata['device_info']['SampleFrequency']), 
                    ch_names = ch_metadata['name'], 
                    units=ch_metadata['unit'])

# Make metadata for the aux channels
name = []
type = []
unit = []
description = []

for i in np.arange(len(metadata['aux_info'])):
    name.append('AUX' + str(i+1))
    type.append('Torque')
    unit.append(metadata['aux_info'][i]['unity_of_measurement'])
    description.append(metadata['aux_info'][i]['description'])

aux_ch_metadata = {'name': name, 'type': type, 'unit': unit, 'description': description}     

# Get a BIDS compatible path and filename
path = bids_gen.datapath
name = bids_gen.subject + '_' + bids_gen.task + '_' + 'torque'

# Convert metadata into a pandas data frame and save tsv-file
df = pd.DataFrame(data=aux_ch_metadata)
df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)


# fsamp = 2048
# # Get duration of the signal in seconds
# seconds = np.ceil(data.shape[0]/fsamp)
# # Add zeros to the signal such that the total length is in full seconds
# signal = np.zeros([int(seconds*fsamp), data.shape[1]])
# signal[0:data.shape[0],:] = data

# edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp, label=ch_names[0])])

# for i in np.arange(1,signal.shape[1]):
#     new_signal = EdfSignal(signal[:,i], 
#                             sampling_frequency=fsamp, 
#                             label=ch_names[i],
#                             physical_dimension=units[i])
# edf.append_signals(new_signal)

# Get a BIDS compatible path and filename
#path = self.datapath  
#name = self.subject + '_' + self.task + '_' + self.datatype

#edf.write(path + name + '.edf')







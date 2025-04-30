import numpy as np
import pandas as pd
from edfio import *
from data2bids import *
from otb_io import open_otb, format_otb_channel_metadata, format_subject_metadata
from sidecar_templates import emg_sidecar_template, dataset_sidecar_template

# Initalize bids dataset, add some metadata to it and save to file
my_dataset = bids_dataset(datasetname = 'my-bids-project')
my_dataset.set_metadata(field_name='dataset_sidecar', source={'Description': 'Some random test dataset'})
my_dataset.write()

# Make a recording that belongs to my_dataset
my_recording = bids_emg_recording(subject=1, data_obj=my_dataset)

# Add some non-sense data to it
some_data = np.random.randn(10,3)
my_recording.set_data(field_name='emg_data',mydata=some_data,fsamp=1)
# Add some metadata
subject_info = {'name': [my_recording.subject_name], 'age': [43]}
my_recording.set_metadata(field_name='subjects_data', source=subject_info)

channel_info = {'name': ['1', '2', '3'], 
                'type': ['EMG', 'EMG', 'EMG'], 
                'unit': ['V', 'V', 'V']}
my_recording.set_metadata(field_name='channels',source=channel_info)

# Save results
my_recording.write()

# Make an other bids dataset by loading what we have just generated
another_bids_dataset = bids_emg_recording(datasetname = 'my-bids-project')
another_bids_dataset.read()
 
print(another_bids_dataset.channels)
print(another_bids_dataset.dataset_sidecar)
print(another_bids_dataset.subjects_data)

print('Finished playing with emg_bids_io class')

# Now let's have a look at the derivative I/O handling
bids_derivative_data = bids_decomp_derivatives()

# Make data of two pseudo sources
my_sources = np.random.randn(10,2)
my_spikes = {'1': [3, 6], '2': [2, 8]}
bids_derivative_data.set_data(field_name='source', mydata=my_sources, fsamp=1)
bids_derivative_data.add_spikes(spikes=my_spikes)

# And also add some metadata to your pipeline
pipeline_info = {'PipelineParameters': {'ext_factor': 12, 'whitening_method': 'ZCA'}}
bids_derivative_data.set_metadata(field_name='pipeline_sidecar', source=pipeline_info)
bids_derivative_data.write()








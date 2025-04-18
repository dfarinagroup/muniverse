import numpy as np
import pandas as pd
from edfio import *
from data2bids import emg_bids_io, decomp_derivatives_bids_io
from otb_io import open_otb, format_otb_channel_metadata, format_subject_metadata
from sidecar_templates import emg_sidecar_template, dataset_sidecar_template

# Initalize bids dataset
bids_dataset = emg_bids_io(subject=1)
# Add some non-sense to it
some_data = np.random.randn(10,3)
bids_dataset.set_raw_data(mydata=some_data,fsamp=1)
# Add some metadata
subject_info = {'name': bids_dataset.subject_id, 'age': 43}
bids_dataset.add_subject_metadata(subject_info)

channel_info = {'name': ['1', '2', '3'], 
                'type': ['EMG', 'EMG', 'EMG'], 
                'unit': ['V', 'V', 'V']}
bids_dataset.add_channel_metadata(channel_info)

dataset_info = {'Name': 'Just a simple toy dataset'}
bids_dataset.add_dataset_sidecar_metadata(dataset_info)

# Save results
bids_dataset.write()

# Make an other bids dataset by loading what we have just generated
another_bids_dataset = emg_bids_io(subject=1)
another_bids_dataset.read()

print(another_bids_dataset.channels)
print(another_bids_dataset.dataset_sidecar)
print(another_bids_dataset.subject)

print('Finished playing with emg_bids_io class')

# Now let's have a look at the derivative I/O handling
derivative_dataset = decomp_derivatives_bids_io()

# Make data of two pseudo sources
two_sources = np.random.randn(10,2)
spikes = {'1': [3, 6], '2': [2, 8]}
derivative_dataset.set_source_data(mysources=two_sources, fsamp=1)
derivative_dataset.add_spikes(spikes=spikes)

# And also add some metadata to your pipeline
pipeline_info = {'PipelineParameters': {'ext_factor': 12, 'whitening_method': 'ZCA'}}
derivative_dataset.add_dataset_sidecar_metadata(pipeline_info)
derivative_dataset.write()








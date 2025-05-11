import numpy as np
import matplotlib.pyplot as plt
from .decomposition_methods import upper_bound, basic_cBSS
from .decomposition_routines import *
from ..evaluation.evaluate import *
from ..data_preparation.data2bids import *
from ..data_preparation.otb_io import open_otb
from pathlib import Path
import time

#BIDS_dataset = bids_dataset(datasetname='Caillet_et_al_2023', root=str(Path.home()) + '/Downloads/')
BIDS_dataset = bids_dataset(datasetname='Avrillon_et_al_2024', root=str(Path.home()) + '/Downloads/')
#BIDS_dataset = bids_dataset(datasetname='neuromotion-test', root=str(Path.home()) + '/Downloads/')
df = BIDS_dataset.list_all_file('emg.edf')
filt_df = df[df['task'].str.contains('20percentmvc')]
emg_recording = bids_emg_recording(data_obj=BIDS_dataset,subject=1, datatype='emg')
emg_recording.read_data_frame(df, 41)
emg_recording.read()

#idx = np.arange(emg_recording.emg_data.num_signals)
idx = np.arange(64)
SIG = edf_to_numpy(emg_recording.emg_data,idx).T
#fsamp = float(emg_recording.channels.loc[0, 'sampling_frequency'])

fsamp = 2048
cBSS = basic_cBSS(ext_fact = 16, opt_initalization = 'random')

# Do the decomposition and report the runtime
start = time.time()
sources, spikes, sil, mu_filters = cBSS.decompose(SIG[idx,20000:60000], fsamp)
end = time.time()
print(f"Runtime: {end - start:.6f} seconds")



print('done')

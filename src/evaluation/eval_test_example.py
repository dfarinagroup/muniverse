import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from .evaluate import *


datasetname = 'Grison_et_al_2025'
pipelinename = 'SCD'
bids_derivative_root = str(Path.home()) + '/Downloads/bids_derivatives/'

df1 = pd.DataFrame({
    'source_id': [1]*5 + [2]*6 + [3]*3,
    'spike_time': np.array([11, 21, 31, 41, 51, 16, 29, 42, 55, 68, 81, 200, 400, 800])*1e-3
})

df2 = pd.DataFrame({
    'source_id': [10]*6 + [20]*5,
    'spike_time': np.array([10, 20, 26, 30, 40, 50, 18, 31, 44, 70, 83])*1e-3
})

out = evaluate_spike_matches(df1, df2, t_end = 1, max_shift = 0.1, )

frames = [df1, df2]
df3 = pd.concat(frames)
labels, mat = label_sources(df3, threshold=0.35, t_end = 1)
print(out.head())
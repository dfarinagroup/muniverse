"""
Quick script to explore reading otb+ and otb4 files via the unified read_otb interface.
Run from the project root:  .venv/bin/python scripts/explore_otb_reading.py
"""

import sys
import tarfile
import tempfile
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from muniverse.utils._otb_io import read_otb

DIVIDER = "─" * 60

def print_section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ── otb+ ──────────────────────────────────────────────────────────────────────

OTB_PLUS_FILE = "/Users/pm1222/Work/data/emg_datasets/TA_larger-denser/RAW_HDEMG_SIGNALS/S1/S1_30MVC.otb+"
print_section(f"OTB+  │  {OTB_PLUS_FILE}")
data_plus, ch_plus = read_otb(OTB_PLUS_FILE)

fs = ch_plus["sampling_frequency"].iloc[0]
n_ch, n_samp = data_plus.shape
print(f"  Shape         : {n_ch} channels × {n_samp} samples  ({n_samp/fs:.1f} s @ {int(fs)} Hz)")
print(f"  Data range    : {data_plus.min():.3f} – {data_plus.max():.3f} mV")
print(f"  channel_info  : {ch_plus.shape[0]} rows, columns: {list(ch_plus.columns)}")
print()
print(f"  Type counts   : {ch_plus['type'].value_counts().to_dict()}")
print(f"  Unit counts   : {ch_plus['units'].value_counts().to_dict()}")
print()
print("  EMG channels (first 3):")
print(ch_plus[ch_plus["type"] == "EMG"].head(3).to_string(index=False))
print()
print("  MISC channels:")
print(ch_plus[ch_plus["type"] == "MISC"].to_string(index=False))


# ── otb4 ──────────────────────────────────────────────────────────────────────

OTB4_FILE = "/Users/pm1222/Work/data/emg_datasets/emanka_tms_emg/pilot1_em_flexor/isometric30pcmvc_2runs.otb4"
print_section(f"OTB4  │  {OTB4_FILE}")
data4, ch4 = read_otb(OTB4_FILE)

print(f"  data shape    : {data4.shape}")
fs4 = ch4["sampling_frequency"].iloc[0]
n_ch4, n_samp4 = data4.shape
print(f"  Shape         : {n_ch4} channels × {n_samp4} samples  ({n_samp4/fs4:.1f} s @ {int(fs4)} Hz)")
print(f"  Data range    : {data4.min():.4f} – {data4.max():.4f}")
print(f"  channel_info  : {ch4.shape[0]} rows, columns: {list(ch4.columns)}")
print()
print(f"  Type counts   : {ch4['type'].value_counts().to_dict()}")
print(f"  Unit counts   : {ch4['units'].value_counts().to_dict()}")
print()
print("  EMG channels (first 3):")
print(ch4[ch4["type"] == "EMG"].head(3).to_string(index=False))
print()
print("  MISC channels:")
print(ch4[ch4["type"] == "MISC"].to_string(index=False))

print()

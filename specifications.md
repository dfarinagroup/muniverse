# **MUniverse: Benchmarking Motor Unit Decomposition Algorithms**

MUniverse is a modular framework for **simulated and experimental EMG dataset generation**, **motor unit decomposition algorithm benchmarking**, and **performance evaluation**. It integrates Biomechanical simulation (via *NeuroMotion*), generative models (*BioMime*), standardized formats (e.g. BIDS/Croissant), and FAIR data hosting (Harvard Dataverse).

---

## From a User’s Perspective

### **Task 1: Generate New Data**
```python
from muniverse.datasets import generate_recording, set_config

neuromotion_config = json.load('../configs/neuromotion_config.json')
recording_config = {'movement_type': 'isometric'}
recording_config = set_config(recording_config, neuromotion_config)
recording, metadata = generate_recording(recording_config)
```

### **Task 2: Use Existing Dataset + Run Algorithm**
```python
from muniverse.datasets import load_dataset
from muniverse.algorithms import decompose
from muniverse.evals import generate_report_card

neuromotion_tiny = load_dataset('../datasets/neuromotion_tiny_croissant.json')
emg, labels, configs = neuromotion_tiny[0]
spikes = decompose(emg, method='scd')

report_card = generate_report_card(spikes, labels, verbosity=0)
```

---

## From a Developer’s Perspective

### **Task 1: Generate Datasets**
Workflow to automate dataset generation across parameter spaces (e.g., LHS or hierarchical sampling):

```python
from muniverse.datasets import generate_recording
<generate a folder of configs>
<call ./scripts/generate_neuromotion_datasets.py for each config>
```

Under the hood:
- `generate_recording(config)` wraps a `run.sh` call, which uses **Docker or Singularity** to launch `run_neuromotion.py` inside a container .
- `run_neuromotion.py` triggers simulation from movement → spike trains → EMG via **NeuroMotion + BioMime**.
- Outputs are saved in BIDS-compliant format and uploaded to Harvard Dataverse using:
```python
convert_to_BIDS_simulatedEMG(recording, metadata)
push_to_dataverse([recording, metadata])
```

### **Task 2: Run Algorithm + Generate Evaluation Report**
```python
from muniverse.datasets import load_dataset
from muniverse.evals import generate_report_card

dataset = load_dataset(<croissant file or dataverse doi>)
recording = load_recording(<recording identifier>)
spikes, process_metadata = decompose(recording)

report_card = generate_report_card(spikes, dataset)
```

To publish:
```python
convert_to_BIDS_derivatives(spikes, type='predictions')
convert_to_BIDS_derivatives(process_metadata, type='process_metadata')
convert_to_BIDS_derivatives(report_card, type='report_card')

push_to_dataverse([spikes, process_metadata, report_card])
```

### **Task 3: Aggregate + Analyze Results**
```python
import pandas as pd

report_cards = load_all_report_cards(<folder or registry>)
df = pd.concat(report_cards, axis=0)
<analyze performance across dataset, method, noise level, etc.>
```

## Current Directory Structure
```
.
├── __init__.py
├── configs
├── data
├── docker
├── docs
├── environment
├── notebooks
├── readme.md
├── sandbox.ipynb
├── scripts
│   ├── __init__.py
│   └── evaluate_recording.py
├── specifications.md
├── src
│   ├── __init__.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── _run_scd.py
│   │   ├── docker/Dockerfile
│   │   ├── emg_snippet.npy
│   │   ├── run_scd.sh
│   │   └── run.py
│   ├── data_generation
│   │   ├── __init__.py
│   │   ├── _run_neuromotion.py 
│   │   ├── config_test.yml
│   │   ├── docker/Dockerfile
│   │   ├── generate_data.py
│   │   └── run.sh
│   ├── data_preparation
│   │   ├── __init__.py
│   │   ├── convert_to_BIDS_experimentalEMG_simple.py
│   │   ├── convert_to_BIDS_simulatedEMG.py
│   │   ├── data2bids.py
│   │   └── sidecar_templates.py
│   ├── datasets
│   │   └── __init__.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── report_card.py
│   ├── registry.py
│   └── utils
│       ├── __init__.py
│       ├── containers.py
│       ├── otb2bids.py
│       └── sub.json
└── tests
    ├── __init__.py
    ├── test_algorithms.py
    ├── test_datasets.py
    └── test_evaluation.py
```

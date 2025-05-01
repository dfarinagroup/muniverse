# **MUniverse: Benchmarking Motor Unit Decomposition Algorithms**

MUniverse is a modular framework for **simulated and experimental EMG dataset generation**, **motor unit decomposition algorithm benchmarking**, and **performance evaluation**. It integrates Biomechanical simulation (via *NeuroMotion*), generative models (*BioMime*), standardized formats (e.g. BIDS/Croissant), and FAIR data hosting (Harvard Dataverse).

## Development Setup

### Package Installation
MUniverse is distributed as a Python package. To set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/your-username/muniverse.git
cd muniverse
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
# Install core dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Project Structure
```
muniverse/
├── src/                    # Package source code
│   ├── __init__.py
│   ├── data_generation/    # Data generation utilities
│   ├── algorithms/         # Decomposition algorithms
│   ├── evaluation/         # Performance evaluation
│   └── utils/              # Utility functions
├── notebooks/              # Tutorial notebooks
├── tests/
├── pyproject.toml
└── README.md
```

---

## User API

### **1: Generate New Data**
```python
from muniverse.data_generation import init, generate_recording

init()
neuromotion_config = json.load('../configs/neuromotion_config.json')
recording_config = {'movement_type': 'isometric'}
recording_config = set_config(recording_config, neuromotion_config)
generate_recording(recording_config)
```

### **2: Use Existing Dataset + Run Algorithm**
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

## Developer API

### **1: Generate Datasets**
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

### **2: Run Algorithm + Generate Evaluation Report**
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

### **3: Aggregate + Analyze Results**
```python
import pandas as pd

report_cards = load_all_report_cards(<folder or registry>)
df = pd.concat(report_cards, axis=0)
<analyze performance across dataset, method, noise level, etc.>
```

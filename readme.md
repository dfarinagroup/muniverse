# **MUniverse: Benchmarking Motor Unit Decomposition Algorithms**

<!-- [![Coverage](https://img.shields.io/badge/coverage-9%25-red)](https://github.com/pranavm19/muniverse) -->

MUniverse is a modular framework for **simulated and experimental EMG dataset generation**, **motor unit decomposition algorithm benchmarking**, and **performance evaluation**. It integrates Biomechanical simulation (via *NeuroMotion*), generative models (*BioMime*), standardized formats (e.g. BIDS/Croissant), and FAIR data hosting (Harvard Dataverse).

## Development Setup

### Package Installation
MUniverse is distributed as a Python package. To set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/pranavm19/muniverse.git
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
│   ├── datasets/           # Dataset loading utilities
│   ├── algorithms/         # Decomposition algorithms
│   ├── evaluation/         # Performance evaluation
│   └── utils/              # Utility functions
├── notebooks/              # Tutorial notebooks
├── scripts/                # Utility scripts
├── tests/
├── pyproject.toml
└── README.md
```

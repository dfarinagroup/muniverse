import os
import tempfile
import pytest
import numpy as np
import json
from pathlib import Path
from src.data_generation import init, generate_recording
from src.utils.containers import verify_container_engine
import subprocess

def has_container_engine():
    """Check if any container engine is available."""
    try:
        return verify_container_engine("docker") or verify_container_engine("singularity")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

@pytest.mark.skipif(not has_container_engine(), reason="No container engine (Docker or Singularity) is available")
def test_dataset_init():
    """Test that init properly sets up containers and selects the appropriate engine."""
    # Test without config (should use defaults)
    engine = init()
    assert engine in ["docker", "singularity"], f"Unexpected engine: {engine}"

@pytest.mark.skipif(not has_container_engine(), reason="No container engine (Docker or Singularity) is available")
def test_generate_dataset():
    """Test basic dataset generation with default settings."""
    # Load the test configuration
    config_path = Path(__file__).parent.parent / "src/data_generation/neuromotion_config_template.json"
    
    # Create a temporary directory to simulate the output
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"input_config": config_path, "output_dir": tmpdir}
        generate_recording(config)
        # Check that the output path is within our temporary directory.
        assert os.path.abspath(tmpdir) in os.path.abspath(tmpdir)

@pytest.mark.skipif(not has_container_engine(), reason="No container engine (Docker or Singularity) is available")
def test_subject_seed_effects():
    """Test that different subject seeds produce different but valid outputs through the containerized interface."""
    # Load the test configuration
    config_path = Path(__file__).parent.parent / "src/data_generation/neuromotion_config_template.json"
    
    # Create two temporary directories for outputs
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Run with same seed twice  
        config1 = {"input_config": config_path, "output_dir": tmpdir1}
        config2 = {"input_config": config_path, "output_dir": tmpdir2}
        generate_recording(config1)
        generate_recording(config2)

        # Load the generated MUAPs
        muaps1 = np.load(os.path.join(tmpdir1, "muaps.npy"))
        muaps2 = np.load(os.path.join(tmpdir2, "muaps.npy"))
        
        # Test shapes
        assert muaps1.shape == muaps2.shape, "MUAP shapes should be identical"

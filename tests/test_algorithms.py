import os
import tempfile

import pytest

from src.algorithms import run

pytestmark = pytest.mark.skip(reason="Algorithm tests not implemented yet")


def test_run_algorithm():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "engine": "docker",
            "script_path": "src/algorithms/run.sh",
            "input_path": "data/sample_input.npy",  # This should be a test fixture file
            "output_dir": tmpdir,
            "python_script": "src/algorithms/run_scd.py",
        }
        metadata = run.run_algorithm("scd", config)
        assert "output_dir" in metadata
        assert tmpdir in metadata["output_dir"]

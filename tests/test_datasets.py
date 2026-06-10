"""
Tests for the datasets module.

These tests require a container engine (Docker or Singularity) to run.
"""

import json
import subprocess
from pathlib import Path
from importlib.resources import files

import numpy as np
import pytest

from muniverse.datasets import generate_synthetic_recording
from muniverse.datasets import init as new_init
from muniverse.utils.containers import verify_container_engine, pull_container, get_container_ref

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# CONFIGS_DIR = Path(__file__).parent.parent / "configs"
# DEFAULT_CONFIG_PATH = CONFIGS_DIR / "neuromotion.json"
DEFAULT_CONFIG_PATH = files("muniverse").joinpath("configs/neuromotion.json")

# Lightweight config override: short movement so the container finishes quickly.
_SHORT_MOVEMENT_OVERRIDE = {
    "EffortProfile": "Trapezoid",
    "AngleProfile": "Constant",
    "InitialAngle": 0,
    "TargetAngle": 0,
    "TargetEffort": 10,
    "NRepetitions": 1,
    "RestDuration": 0.5,
    "RampDuration": 0.5,
    "HoldDuration": 0.5,
    "SinFrequency": 0,
    "MovementDuration": 2,
}

_SHORT_DOF = "Test"

DOCKER_IMAGE = "pranavm19/muniverse:neuromotion"


def _pick_engine():
    """Return the first available container engine, or None."""
    for engine in ("singularity", "docker"):
        try:
            if verify_container_engine(engine):
                return engine
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return None


def _load_short_config():
    """Load default config with a short movement profile for fast testing."""
    with open(DEFAULT_CONFIG_PATH) as f:
        config = json.load(f)
    config["MovementConfiguration"]["MovementDOF"] = _SHORT_DOF
    config["MovementConfiguration"]["MovementProfileParameters"].update(
        _SHORT_MOVEMENT_OVERRIDE
    )
    return config


def _has_container():
    return _pick_engine() is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = _pick_engine()
    if e is None:
        pytest.skip("No container engine available")
    return e


@pytest.fixture(scope="module")
def container(engine):
    """Pull the container image if needed and return the engine-appropriate reference."""
    pull_container(DOCKER_IMAGE, engine)
    return get_container_ref(DOCKER_IMAGE, engine)


@pytest.fixture(scope="module")
def short_config(tmp_path_factory):
    """Write the short config to a temp file and return the path."""
    cfg = _load_short_config()
    p = tmp_path_factory.mktemp("cfg") / "test_config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


# ---------------------------------------------------------------------------
# New datasets API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_container(), reason="No container engine available")
def test_new_generate_synthetic_recording_structure(engine, container, short_config, tmp_path):
    """New API returns a dict with the expected keys and shapes."""
    results = generate_synthetic_recording(
        input_config=short_config,
        output_dir=str(tmp_path),
        engine=engine,
        container=container,
        cache_dir=None,
        verbose=True
    )

    expected_keys = {"emg", "spikes", "effort_profile", "angle_profile", "muaps", "muap_angle_labels"}
    assert expected_keys.issubset(results.keys()), (
        f"Missing keys: {expected_keys - results.keys()}"
    )

    emg = results["emg"]
    assert emg.ndim == 3, f"EMG should be 3-D (rows, cols, time), got shape {emg.shape}"

    # Verify output file was saved
    assert (tmp_path / "emg_data.npz").exists()


@pytest.mark.skipif(not _has_container(), reason="No container engine available")
def test_new_generate_synthetic_recording_determinism(engine, container, short_config, tmp_path):
    """Same config+seed produces identical EMG output on two runs."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    r1 = generate_synthetic_recording(
        input_config=short_config,
        output_dir=str(out1),
        engine=engine,
        container=container,
        cache_dir=None,
    )
    r2 = generate_synthetic_recording(
        input_config=short_config,
        output_dir=str(out2),
        engine=engine,
        container=container,
        cache_dir=None,
    )

    np.testing.assert_array_equal(
        r1["emg"], r2["emg"],
        err_msg="EMG output is not deterministic across two runs with the same seed",
    )


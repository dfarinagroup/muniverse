"""
Simple tests of the algorithms

"""

import numpy as np
import pickle

from dataclasses import dataclass

from muniverse.algorithms.cbss import FastIcaCBSS
from muniverse.algorithms.ae_decomposer import AEDecoder

from muniverse.algorithms import decompose_recording
from muniverse.algorithms.decomposition import load_config

# Some helper functions to make some simple test signal
@dataclass
class MU:
    waveform: np.ndarray
    recruitment_idx: int
    isi: int

def _generate_spikes(rng, start, step, n, jitter):
    base = np.arange(start, start + n * step, step)
    noise = rng.integers(-jitter, jitter + 1, size=n)
    return base + noise

def _insert_mu(n_samples, spikes, waveform):
    signal = np.zeros(n_samples)
    half = len(waveform) // 2
    for spike in spikes:
        samples = slice(spike - half, spike + half + 1)
        signal[samples] += waveform
    return signal    

def _make_fake_emg(
    n_channels=3,
    n_samples=1000,
    n_spikes=80,
    jitter=2,
    seed=42,
):

    rng = np.random.default_rng(seed)

    emg = rng.standard_normal((n_channels, n_samples))

    mus = [
        MU(
            waveform=np.array([-3.33, 10, -3.33]),
            recruitment_idx=10,
            isi=11,
        ),
        MU(
            waveform=np.array([3, -9, 3]),
            recruitment_idx=5,
            isi=12,
        ),
        MU(
            waveform=np.array([-2, -8, 10]),
            recruitment_idx=7,
            isi=10,
        ),
    ]

    for ch, mu in enumerate(mus):

        spikes = _generate_spikes(
            rng=rng,
            start=mu.recruitment_idx,
            step=mu.isi,
            n=n_spikes,
            jitter=jitter,
        )

        mu_sig = _insert_mu(
            n_samples=n_samples,
            spikes=spikes,
            waveform=mu.waveform,
        )

        emg[ch] += emg[ch] + mu_sig

    return emg   

def test_cbss_simple():
    """ Test the CBSS algorithm using simple fake data """

    # Generate test data
    data = _make_fake_emg()
    fsamp = 100

    # Set parameters
    cfg = {
        "ext_fact": 3,
        "ica_iterations": 3,
        "ica_tol": 5e-2,
    }

    # Load reference results
    file = "./expected_results/simple_cbss_prediction.pkl"
    with open(file, "rb") as f:
        ref_results = pickle.load(f)

    try:
        model = FastIcaCBSS(**cfg)
        spikes, sources, scores = model.fit_predict(data,fsamp)
        return_code = 0
    except:
        return_code = 1

    assert return_code == 0, "CBSS failed"
    assert model.ext_fact == 3, "wrong parameter extension factor"
    assert spikes.equals(ref_results["spikes"]), (
        "Predicted spikes deviate from the reference results"
    )
    assert np.isclose(sources, ref_results["sources"]).all(), (
        "Predicted sources deviate from the reference results"
    )
    assert np.isclose(scores["sil"], ref_results["scores"]["sil"]).all(), (
        "Predicted sil scores deviate from the reference results"
    )

def test_ae_simple():
    """ Test the autoencoder algorithm using simple fake data """

    # Generate test data
    data = _make_fake_emg()
    fsamp = 100

    # Define Algorithm Parameters
    cfg = {
        "ext_fact": 3,
        "latent_dim": 3
    }

    # Load reference results
    file = "./expected_results/simple_ae_prediction.pkl"
    with open(file, "rb") as f:
        ref_results = pickle.load(f)

    try:
        model = AEDecoder(**cfg)
        spikes, sources, scores = model.fit_predict(data,fsamp)
        return_code = 0
    except:
        return_code = 1

    assert return_code == 0, "CBSS failed"
    assert model.ext_fact == 3, "wrong extension factor"   
    assert spikes.equals(ref_results["spikes"]), (
        "Predicted spikes deviate from the reference results"
    )
    assert np.isclose(sources, ref_results["sources"]).all(), (
        "Predicted sources deviate from the reference results"
    )
    assert np.isclose(scores["sil"], ref_results["scores"]["sil"]).all(), (
        "Predicted sil scores deviate from the reference results"
    )

def test_decompose_recording():  

    # Generate test data
    data = _make_fake_emg()
    fsamp = 100

    # Pre-processing config
    pre_cfg = [
        {
            "step": "bandpass",
            "high_pass": 3,
            "low_pass": 30,
            "method": "butter",
            "order": 2
        },
        {
            "step": "time_window",
            "t_start": 0,
            "t_end": -1
        }
    ]
    # Post-processing config
    post_cfg = [
        {
            "step": "remove_duplicates",
            "max_shift": 0.01,
            "tolerance": 0.001,
            "threshold": 0.3,
            "quality_metric": "sil",
            "mode": "max"
        },
        {
            "step": "bad_source_detection",
            "quality_metric": "sil",
            "threshold": 0.9,
            "min_spikes": 10,
            "mode": "below"
        }  
    ]

    # Run a FastIcaCBSS Pipeline
    cfg = {
        "preProcessingConfig": pre_cfg,
        "algorithmConfig": {
            "ext_fact": 3,
            "ica_iterations": 3,
            "ica_tol": 5e-2,
        },
        "postProcessingConfig": post_cfg
    }

    _, log = decompose_recording(
        data=data, fsamp=fsamp, method="cbss", algorithm_config=cfg
    )

    return_codes = log["Execution"]["ReturnCodes"]

    assert return_codes["PreProcessEMG"] == 0, "Pre-processing failed"
    assert return_codes["FastIcaCBSS"] == 0, "FastIcaCBSS failed"
    assert return_codes["PostProcessSpikes"] == 0, "Post-processing failed"  

    # Run AE Pipeline
    cfg = {
        "preProcessingConfig": pre_cfg,
        "algorithmConfig": {
            "ext_fact": 3,
            "latent_dim": 3
        },
        "postProcessingConfig": post_cfg
    }

    _, log = decompose_recording(
        data=data, fsamp=fsamp, method="ae", algorithm_config=cfg
    )

    return_codes = log["Execution"]["ReturnCodes"]

    assert return_codes["PreProcessEMG"] == 0, "Pre-processing failed"
    assert return_codes["AEDecoder"] == 0, "AEDecoder failed"
    assert return_codes["PostProcessSpikes"] == 0, "Post-processing failed"   

    try:
        import scd as scd
        import torch

        cfg = {
            "preProcessingConfig": [],
            "algorithmConfig": {
                "sampling_frequency": fsamp,
                "low_pass_cutoff": 30,
                "high_pass_cutoff": 3,
                "extension_factor": 3,
                "max_iterations": 3,
            },
            "postProcessingConfig": []
        }

        _, log = decompose_recording(
            data=data, fsamp=fsamp, method="scd", engine="local", algorithm_config=cfg
        )

        return_codes = log["Execution"]["ReturnCodes"]

        assert return_codes["PreProcessEMG"] == 0, "Pre-processing failed"
        assert return_codes["scd"] == 0, "SCD (local engine) failed"
        assert return_codes["PostProcessSpikes"] == 0, "Post-processing failed" 


    except ImportError:
        scd is None





    
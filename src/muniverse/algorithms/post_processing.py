import warnings
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .cbss import _BaseCBSS
from .core import (
    est_spike_times,
    find_outliers,
    spike_dict_to_long_df,
    get_duplicates_mask, 
    get_bad_source_mask,
    spike_dict_to_long_df,
    filter_spikes
)    
from ..evaluation.evaluate import (
    pseudo_sil_score
)

class PostProcessSpikes:
    """
    Class to post process motor unit spike trains.

    Parameters
    ----------
    steps : list of dict
        List of post processing steps. Each step is a dictionary describing
        the processing operation.

        Supported step types are:
        "remove_dublicates", "bad_source_detection", "mask_sources"

        **Remove Duplicates**: Automatically detect duplicates in your sources::

            {
                "step": "remove_duplicates",
                "max_shift": float,
                "tolerance": float,
                "threshold": float,
                "quality_metric": str,
                "mode": "max" | "min"
            }

        **Bad Source Detection**: Automatically detect bad sources based on a quality
        metric and some theshold::  

            {
                "step": "bad_source_detection",
                "quality_metric": str,
                "threshold_value": float,
                "min_spikes": int,
                "mode": "below" | "above" 
            } 

        **Mask Sources**: Mask all sources given in "sources_list" to be excluded in the following. 
        Can be either used to reject known bad sources or limit the analysis to a subset of your data::  

            {
                "step": "mask_sources",
                "source_list": list[int]
            }

    Examples:
    ---------
    Post decomposition outputs by removing duplicates and rejecting bad sources.
    >>> model = post_processing(steps = [
    ...     {
    ...         "step": "remove_duplicates",
    ...         "max_shift": 0.01,
    ...         "tolerance": 0.001,
    ...         "threshold": "0.3",
    ...         "quality_metric": "sil",
    ...         "mode": "max"
    ...     },
    ...     {
    ...         "step": "bad_source_detection",
    ...         "quality_metric": "sil",
    ...         "threshold": 0.9,
    ...         "min_spikes": 10,
    ...         "mode": "below"
    ...     },
    ... ])
    >>> out = model.post_process(...)                     

    """

    def __init__(
            self, 
            steps: list[dict] = [
                {  
                    "step": "remove_duplicates",
                    "max_shift": 0.01,
                    "tolerance": 0.001,
                    "theshold": 0.3,
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
    ):

        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class RemoveDuplicates(BaseModel):
        step: Literal["remove_duplicates"]
        max_shift: float = 0.01
        tolerance: float = 0.001
        threshold: float = 0.3
        quality_metric: str = "sil"
        window: tuple[float, float] = (0, -1)
        mode: Literal["max", "min"] = "max"
        description: str = "Duplicate source"

    class BadSourceDetection(BaseModel):
        step: Literal["bad_source_detection"]
        quality_metric: str = "sil"
        threshold: float = 0.9
        min_spikes: int = 10
        mode: Literal["above", "below"] = "below" 
        description: str = "Below quality threshold"

    class MaskSources(BaseModel):
        step: Literal["mask_sources"]
        unit_ids: list[int] = []  
        description: str = "Manually masked"      
                          
    PostProcessStep = Annotated[
        Union[
            RemoveDuplicates, 
            BadSourceDetection,
            MaskSources
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PostProcessStep)

    def add_step(self, step):
        """ Add an additional post processing step"""
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

    def _apply_base_step(
        self,
        step,
        spikes,
        sources,
        scores,
        fsamp,
        unit_status,
        source_mask
    ):

        if isinstance(step, self.RemoveDuplicates):

            if not step.quality_metric in scores.keys():
                raise ValueError(
                    f"The slected qaulity metric {step.quality_metric} is not defined"
                )

            local_mask, new_labels = get_duplicates_mask(
                spikes=spikes,
                scores=scores[step.quality_metric],
                fsamp=fsamp,
                mode=step.mode,
                t_start=step.window[0],
                t_end=step.window[1],
                duplicate_theshold=step.threshold,
                max_shift=step.max_shift,
                tol=step.tolerance,
                mask=source_mask
            )
            source_mask = source_mask & local_mask
            unit_status.loc[~local_mask, "status"] = "masked"
            unit_status.loc[
                ~local_mask, "status_description"
            ] = step.description
            unit_status["unit_id"] = new_labels.astype(int)

        elif isinstance(step, self.BadSourceDetection):

            if not step.quality_metric in scores.keys():
                raise ValueError(
                    f"The slected qaulity metric {step.quality_metric} is not defined"
                )

            local_mask = get_bad_source_mask(
                spikes=spikes,
                score=scores[step.quality_metric],
                threshold=step.threshold,
                mode=step.mode,
                min_num_spikes=step.min_spikes
            )
            source_mask = source_mask & local_mask
            unit_status.loc[~local_mask, "status"] = "masked"
            unit_status.loc[
                ~local_mask, "status_description"
            ] = step.description
        elif isinstance(step, self.MaskSources):

            n_source = len(spikes["unit_id"].unique())
            local_mask = np.ones(n_source, dtype=bool)
            local_mask[step.unit_ids] = False
            source_mask = source_mask & local_mask
            unit_status.loc[~local_mask, "status"] = "masked"
            unit_status.loc[
                ~local_mask, "status_description"
            ] = step.description

        return source_mask, unit_status   

    def post_process(
            self, 
            spikes: pd.DataFrame, 
            fsamp: float,
            scores: dict | None = None,
            sources: np.ndarray | None = None
    ):
        
        """
        Post process decomposed motor unit spike trains
        specified list of steps.

        Args
        ----
            data : np.ndarray (n_channels, n_samples)
                EMG data 
            spikes : pd.DataFrame
                Table of motor unit spikes    
            fsamp : float 
                Sampling rate in Hz
            scores : dict or None 
                A Dictonary of source quality scores     
            sources : np.ndarray (n_units, n_samples) or None 
                The predicted sources

        Returns
        -------
            spikes : pd.DataFrame
                Table of motor unit spikes
            sources : np.ndarray (n_units, n_samples)
                The predicted sources / latents
            score : dict
                A dictonary of source quality scores    
            metadata : dict
                A dictonary of processing metadata     
        
        """

        # Mask bad sources
        n_units = len(spikes["unit_id"].unique())
        source_mask = np.ones(n_units, dtype=bool)

        unit_status = pd.DataFrame({
            "orig_unit_id": [i for i in range(n_units)],
            "status": ["good"] * n_units,
            "status_description": ["n/a"] * n_units,
            "unit_id": [i for i in range(n_units)]
        })

        if self.steps is not None:
            for step in self.steps:

                source_mask, unit_status = self._apply_base_step(
                    step = step,
                    spikes = spikes,
                    sources = sources,
                    scores = scores,
                    fsamp = fsamp,
                    unit_status= unit_status,
                    source_mask = source_mask
                )

        # Package the applied processing steps
        steps = [step.model_dump() for step in self.steps] 

        metadata = {
            "fsamp": fsamp,
            "source_mask": source_mask,
            "unit_status": unit_status,
            "steps": steps
        }
              
        new_spikes, _ = filter_spikes(spikes, source_mask)
        new_sources = sources[source_mask, :]
        new_scores = {}
        for k, v in scores.items():
            if isinstance(v, np.ndarray) and v.shape[0] == len(source_mask):
                new_scores[k] = v[source_mask]
            else:
                new_scores[k] = v
        
        return new_spikes, new_sources, new_scores, metadata
    

class PostProcessCBSS(_BaseCBSS, PostProcessSpikes):
    """
    
    Class to post process motor unit spike trains and
    integrating the CBSS framework 
    
    Parameters
    ----------
    steps : list of dict
        List of post processing steps. Each step is a dictionary describing
        the processing operation.

        Supported step types are:
        "remove_dublicates", "bad_source_detection", "mask_sources"

        **Remove Duplicates**: Automatically detect duplicates in your sources::

            {
                "step": "remove_duplicates",
                "max_shift": float,
                "tolerance": float,
                "threshold": float,
                "quality_metric": str,
                "mode": "max" | "min"
            }

        **Bad Source Detection**: Automatically detect bad sources based on a quality
        metric and some theshold::  

            {
                "step": "bad_source_detection",
                "quality_metric": str,
                "threshold_value": float,
                "min_spikes": int,
                "mode": "below" | "above" 
            } 

        **Mask Sources**: Mask all sources given in "sources_list" to be excluded in the following. 
        Can be either used to reject known bad sources or limit the analysis to a subset of your data::  

            {
                "step": "mask_sources",
                "source_list": list[int]
            }

        **Predict Spikes**: Use the learned unmixing weights to predict 
        motor unit spikes in the time window specified by t_start and 
        t_end::

            {
                "step": "predict_spikes",
                "t_start": float,
                "t_end": float
            
            }

        **Fit From Spikes**: Supervised learning of  the unmixing weights of a 
        CBSS model given a set of motor unit spike labels (in the specified time 
        window). The learned unmixing weights are then applied to the data::

            {
                "step": "predict_spikes",
                "t_start": float,
                "t_end": float
            
            }        
    
    """

    def __init__(
            self, 
            steps = [],
            ext_fact = 12, 
            whitening_method = "ZCA", 
            whitening_backend: Literal["ed", "svd"] = "ed",
            whitening_reg = "auto", 
            spike_detection_exp = 2, 
            spike_detection_min_delay = 0.01, 
            max_spike_shifts = 0.01,
            verbose = False
        ):
        super().__init__(
            ext_fact = ext_fact, 
            whitening_method = whitening_method, 
            whitening_backend = whitening_backend,
            whitening_reg = whitening_reg, 
            spike_detection_exp = spike_detection_exp, 
            spike_detection_min_delay = spike_detection_min_delay, 
            verbose = verbose
        )

        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]
        self.max_spike_shifts = max_spike_shifts

    class PredictSpikes(BaseModel):
        step: Literal["predict_spikes"]
        rewhiten: bool = True
        t_start: float = 0
        t_end: float = -1 

    class FitFromSpikes(BaseModel):
        step: Literal["fit_from_spikes"]
        t_start: float = 0
        t_end: float = -1    

    PostProcessStep = Annotated[
        Union[
            PostProcessSpikes.RemoveDuplicates, 
            PostProcessSpikes.BadSourceDetection,
            PostProcessSpikes.MaskSources,
            PredictSpikes,
            FitFromSpikes,
        ],
        Field(discriminator="step")
    ]

    _adapter = TypeAdapter(PostProcessStep)
    
    def _get_win_samples(
            self, 
            data: np.ndarray, 
            fsamp: float,
            t_start: float,
            t_end: float
        ):
        """Extract time window samples"""

        duration = (data.shape[1] - 1) / fsamp
        if t_end > duration or t_end == -1:
            t_end = duration
        if t_start < 0:
            t_start = 0    
        t = np.linspace(0, duration, data.shape[1])
        sample_idx = (t >= t_start) & (t <= t_end)

        return sample_idx     

    def rewhiten(self, data):
        """Recalculate the whiening matrix"""

        ext_sig = self._extension(data)
        self._whitening(ext_sig, return_data=False)
                
    def fit_predict_from_spike_labels(
            self, 
            sig: np.ndarray, 
            spikes: pd.DataFrame, 
            fsamp: float, 
        ):
        """
        Supervised fitting of a CBSS model given EMG
        data and motor unit spike labels

        Args
        ----
            sig : np.ndarray (n_channels, n_samples)
                EMG data matrix
            spikes : pd.DataFrame
                Table of motor unit spike labels
            fsamp : float
                Sampling rate in Hz

        Returns
        -------
            spikes : pd.DataFrame 
                Table of motor unit spikes (can be temporally shifted)
            sources : np.ndarray 
                Estimated sources / ica components (n_components, n_samples)     
            scores : dict
                Dictonary of source quality scores            

        
        
        """

        ext_sig = self._extension(sig)

        white_sig = self._whitening(ext_sig)

        units = sorted(spikes["unit_id"].unique())
        n_units = len(units)

        new_spikes = {i: [] for i in range(n_units)}
        scores = {
            "sil": np.zeros(n_units),
            "cov_isi": np.zeros(n_units),
        }
        self.unmixing_weights_ = np.zeros((white_sig.shape[0], n_units))

        sources = np.zeros((n_units, white_sig.shape[1]))

        for i in range(n_units):

            local_spikes = spikes[
                spikes["unit_id"] == i]["sample"].values

            w, new_spikes[i], sil = self._optimze_delay(
                X = white_sig,
                spikes=local_spikes,
                fsamp=fsamp
            )

            sources[i, :] = w @ white_sig
            self.unmixing_weights_[:, i] = w
            scores["sil"][i] = sil
            scores["cov_isi"][i] = self._calc_cov_isi(new_spikes[i], fsamp)

        # Convert dict of spikes to long-formated spike table 
        new_spikes = spike_dict_to_long_df(new_spikes)    

        # Update the unmixing format
        self.unmixing_format_ = "white"

        return new_spikes, sources, scores

    def _optimze_delay(self, X, spikes, fsamp):
        """
        Helper function to find optimal delay for 
        a set of motor unit spike labels
        
        """

        if self.max_spike_shifts == 0:
            delays = [0]
        else:
            max_shift = int(self.max_spike_shifts * fsamp)
            delays = range(-max_shift,max_shift+1)

        W = np.zeros((X.shape[0], len(delays)))
        local_scores = np.zeros(len(delays))   

        for j in delays:

            local_spikes = spikes + delays[j]

            w = np.mean(X[:, local_spikes], axis=1)
            w = w / np.linalg.norm(w)
            local_source = w.T @ X
            local_scores[j], _ = pseudo_sil_score(
                source=local_source,
                spikes=local_spikes,
                fsamp=fsamp,
                min_peak_dist=self.spike_detection_min_delay,
                match_dist=0.001
            )
            W[:, j] = w 

        idx = np.argmax(local_scores)
        w = W[:, idx]
        sil = local_scores[idx]
        new_spikes = spikes + delays[idx]
 
        return w, new_spikes, sil    
    
    def post_process(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            spikes: pd.DataFrame, 
            fsamp: float,
            scores: dict | None = None,
            sources: np.ndarray | None = None, # (n_sources x n_samples)
            unmixing_weights: np.ndarray | None = None,
            whitening_matrix: np.ndarray | None = None, 
            unmixing_format: Literal["white", "extended"] = "white"
    ):
        
        """
        Post process decomposed motor unit spike trains
        specified list of steps.

        Args
        ----
            data : np.ndarray (n_channels x n_samples)
                EMG data 
            spikes : pd.DataFrame
                Lits of motor unit spikes    
            fsamp : float 
                Sampling rate in Hz
            scores : dict or None 
                Dictonary of source quality scores     
            sources : np.ndarray or None (n_sources x n_samples)
                The predicted sources
            unmixing_weights: np.ndarray or None , default None
                Weights of the unmixing matrix 
            whitening_matrix : np.ndarray or None , default None
                Whitening matrix 
            unmixing_format : {"white", "extended"} , default "white"    
                Format in which the unmixing weights are provided


        Returns
        -------
            spikes : pd.DataFrame
                Table of motor unit spikes
            sources : np.ndarray (n_units, n_samples)
                The predicted sources / latents
            score : dict
                A dictonary of source quality scores    
            metadata : dict
                A dictonary of processing metadata     
           
        
        """

        self.unmixing_weights_ = unmixing_weights
        self.whiten_ = whitening_matrix
        self.unmixing_format_ = unmixing_format


        metadata = {}
        metadata["fsamp_out"] = fsamp

        # Mask bad sources
        n_sources = sources.shape[0]
        source_mask = np.ones(n_sources, dtype=bool)

        if self.steps is not None:
            for step in self.steps:
                
                source_mask = self._apply_base_step(
                    step = step,
                    spikes = spikes,
                    sources = sources,
                    scores = scores,
                    fsamp = fsamp,
                    source_mask = source_mask
                )

                if isinstance(step, self.PredictSpikes):

                    sample_idx = self._get_win_samples(
                        data, fsamp, step.t_start, step.t_end
                    )
                    if step.rewhiten:
                        self.rewhiten(data[:, sample_idx])  

                    if unmixing_format == "extended":
                        self.unmixing_weights_ = self.whiten_ @ self.unmixing_weights_
                        self.unmixing_format_ = "white"

                    spikes, sources, local_scores = self.predict(
                        sig=data[:, sample_idx],
                        fsamp=fsamp
                    )
                    scores.update(local_scores)
                    
                elif isinstance(step, self.FitFromSpikes):

                    spikes, sources, local_scores = self.fit_predict_from_spike_labels(
                        sig=data,
                        fsamp=fsamp,
                        spikes=spikes
                    )
                    scores.update(local_scores)    

        new_spikes, label_map = filter_spikes(spikes, source_mask)
        new_sources = sources[source_mask, :]
        new_scores = {
            "sil": scores["sil"][source_mask],
            "cov_isi": scores["cov_isi"][source_mask]
        }
 
        return new_spikes, new_sources, new_scores

        
    




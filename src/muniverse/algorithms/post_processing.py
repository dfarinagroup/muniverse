import warnings
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import welch
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .core import extension, whitening, est_spike_times
from ..evaluation.evaluate import (
    get_bin_spikes, 
    max_xcorr,
    match_spike_trains,
    label_sources 
)


class post_processing:
    """
    Class to preprocess HD-EMG data.

    Parameters
    ----------
    steps : list of dict
        List of post processing steps. Each step is a dictionary describing
        the processing operation.

        Supported step types are:
        "remove_dublicates", "bad_source_detection", "mask_sources"

        max_shift=0.1, tol=0.001, threshold=0.3

        **Remove Duplicates**: Automatically detect duplicates in your sources::

            {
                "step": "remove_duplicates",
                "max_shift": float,
                "tolerance": float,
                "threshold": float,
                "quality_metric": "sil" | "cov_isi",
            }

        **Bad Source Detection**: Automatically detect bad sources::  

            {
                "step": "bad_source_detection",
                "quality_metric": "sil" | "cov_isi",
                "threshold_value": float,
                "min_spikes": int
            } 

        **Mask Sources**: Mask all sources given in "sources_list" to be excluded in the following. 
        Can be either used to reject known bad sources or limit the analysis to a subset of your data::  

            {
                "step": "mask_sources",
                "source_list": list[int]
            }

        **Map time window**: Map time samples from your decomposed segment to the 
        global time of your emg recording. The values of t_start and t_end correspond to
        the start and end of the decomposed segment::  

            {
                "step": "map_time_window",
                "t_start": float,
                "t_end": float 
            }    

    Examples:
    ---------
    Post decomposition outputs by removing duplicates and rejecting bad sources.
    >>> model = post_processing(steps = [
    >>>     {
    >>>         "step": "remove_duplicates",
    >>>         "max_shift": 0.01,
    >>>         "tolerance": 0.001,
    >>>         "threshold": "0.3",
    >>>         "quality_metric": "sil"
    >>>     },
    >>>     {
    >>>         "step": "bad_source_detection",
    >>>         "quality_metric": "sil",
    >>>         "threshold": 0.9,
    >>>         "min_spikes": 10
    >>>     },
    >>> ])
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
                },
                {
                    "step": "bad_source_detection",
                    "quality_metric": "sil",
                    "threshold": 0.9,
                    "min_spikes": 10,
                }    
            ]          
    ):

        #self.pre_process_steps = pre_process_steps
        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class RemoveDuplicates(BaseModel):
        step: Literal["remove_duplicates"]
        max_shift: float = 0.01
        tolerance: float = 0.001
        threshold: float = 0.3
        quality_metric: Literal["sil", "cov_isi"] = "sil"
        window: tuple[float, float] | None = None

    class BadSourceDetection(BaseModel):
        step: Literal["bad_source_detection"]
        quality_metric: Literal["sil", "cov_isi"]
        threshold: float = 0.9
        min_spikes: int = 10 

    class MaskSources(BaseModel):
        step: Literal["mask_sources"]
        unit_ids: list[int] = []

    class ApplyUnmixing(BaseModel):
        step: Literal["apply_unmixing"]
        whiten_data: bool = True
        ext_factor: int = 12 
        t_start: float = 0
        t_end: float = -1 

    class RefineSources(BaseModel):
        step: Literal["refine_sources"]
        whiten_data: bool = True
        ext_factor: int = 12
        refinement_loss: Literal["sil", "cov_isi"] = "sil"
        max_iter: int = 10  
        t_start: float = 0
        t_end: float = -1               
  
    PostProcessStep = Annotated[
        Union[
            RemoveDuplicates, 
            BadSourceDetection,
            MaskSources,
            ApplyUnmixing,
            RefineSources 
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PostProcessStep)

    def add_step(self, step):
        """ Add an additional post processing step"""
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

    def _detect_duplicates(
            self, 
            events,
            scores, 
            fsamp, 
            t_start, 
            t_end,
            threshold,
            max_shift,
            tol,
            keep_mask
    ):
        """
        TODO
        
        """

        if keep_mask is not None:
            idx = np.where(~keep_mask)
            scores[idx] = -1

        units = sorted(events["unit_id"].unique())
        n_source = len(units)
        keep_mask = np.zeros(n_source, dtype=bool)

        new_labels, match_matrix = label_sources(
            events, fsamp=fsamp, t_start=t_start, t_end=t_end, 
            threshold=threshold, max_shift=max_shift, tol=tol
        )

        unique_labels = np.unique(new_labels)

        for label in unique_labels:
            idx = np.where(new_labels == label)[0]

            # pick best according to score
            best_idx = idx[np.argmax(scores[idx])]

            keep_mask[best_idx] = True 

        return keep_mask
            
    def _detect_bad_sources(
            self,
            events,
            score,
            threshold=0.9,
            min_num_spikes=10,
            keep_mask=None,
    ):
        """
        Generate a boolean mask that filters out bad sources based on a quality 
        scor and minimum number of spikes. 

        Args
        ----
            events : pd.DataFrame
                Spike data frame
            score : np.ndarray
                Vector of scores

        Returns
        -------
            keep_mask : np.ndarray
                Boolean mask (n_mu,)
        """

        units = sorted(events["unit_id"].unique())
        n_source = len(score)

        # Initialize mask 
        if keep_mask is None:
            keep_mask = np.ones(n_source, dtype=bool)
        else:
            keep_mask = keep_mask.copy()  # avoid side effects

        # Reject bad sources
        for i in range(n_source):
            if not keep_mask[i]:
                continue  # already rejected

            spikes = events[events["unit_id"] == units[i]]["onset"].values
            n_spikes = len(spikes)        
            if score[i] < threshold or n_spikes < min_num_spikes:
                keep_mask[i] = False

        return keep_mask  

    def _filter_events(
            self, 
            events, 
            keep_mask
    ):
        """
        Filter BIDS spike events using a boolean mask.

        Parameters
        ----------
        events : pd.DataFrame
            BIDS events table. Must contain a column with unit_id labels.
        keep_mask : np.ndarray (bool)
            Boolean mask of shape (n_units,) indicating kept sources.

        Returns
        -------
        filtered_events : pd.DataFrame
            Events with bad units removed and remapped labels 
        label_mapping : dict
            Mapping of from old to new unit_id labels
        """

        events = events.copy()

        # --- keep only valid units ---
        valid_units = np.where(keep_mask)[0]
        events = events[events["unit_id"].isin(valid_units)]

        # --- remap labels to 0..N-1 ---
        unique_units = np.sort(events["unit_id"].unique())

        label_map = {old: new for new, old in enumerate(unique_units)}

        events["unit_id"] = events["unit_id"].map(label_map)

        return events, label_map    

    def _apply_unmxing(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            fsamp: float,
            weights: np.ndarray, # (n_observations x n_sources)
            ext_factor: int,
            whiten: bool = True,
            t_start: float = 0,
            t_end: float = -1,

    ):
        """ 
        TODO 
        
        Args
        ----
            sig : np.ndarray 
                Input (EMG) signal (n_channels x n_samples)
            fsamp : float 
                Sampling frequency in Hz
            weights : np.ndarray 
                Learned weights of the unmixing matrix    
            ext_factor : int
                Extension factor 
            whiten : bool
                If True, reconstruct sources in a whitened space
            t_start : float
                Start time of the considered time window in seconds  
            t_end : float
                End time of the considered time window in seconds                

        Returns
        -------
            sources : np.ndarray 
                Estimated sources / ica components (n_components x n_samples)
            spikes : dict 
                Sample indices of motor neuron discharges
            scores : dict
                Dictonary of source quality scores
        
        """

        duration = (data.shape[1] - 1) / fsamp
        if t_end > duration or t_end == -1:
            t_end = duration
        if t_start < 0:
            t_start = 0    

        t = np.linspace(0, duration, data.shape[1])
        sample_idx = np.argwhere(t > t_start & t < t_end)
        sources = np.zeros(weights.shape[1], data.shape[1])

        n_sources = weights.shape[1]

        ext_sig = extension(data[:, sample_idx], ext_factor)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

        if whiten:
            white_sig, Z = whitening(
                Y=ext_sig, 
                method="ZCA"
            )
            sources[:, sample_idx] = weights.T @ white_sig

        else:
            covariance = ext_sig @ ext_sig.T / (ext_sig.shape[1] - 1)
            sources[:, sample_idx] = (
                weights.T @ np.linalg.pinv(covariance) @ ext_sig
            )

        spikes = {}
        scores = {
            "sil": np.zeros(n_sources),
            "cov_isi": np.zeros(n_sources)
        }
        for i in range(sources.shape[1]):
            spikes[i], scores["sil"][i] = est_spike_times(
                sources[i, :], fsamp, cluster="kmeans",
            ) 
            if len(spikes[i]) > 2:
                isi = np.diff(spikes[i] / fsamp)
                scores["cov_isi"][i] = np.std(isi) / np.mean(isi)
            else:
                scores["cov_isi"][i] = np.inf   

        return spikes, sources, scores 

    def post_process(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            spikes: pd.DataFrame, 
            fsamp: float,
            sources: np.ndarray | None, # (n_sources x n_samples)
            unmixing_weights: np.ndarray | None,
            scores: dict | None,
            
    ):
        
        """
        Post process multi-channel time series data using the 
        specified list of steps.

        Args
        ----
            data : np.ndarray (n_channels x n_samples)
                EMG data 
            spikes : pd.DataFrame
                Lits of motor unit spikes    
            fsamp : float 
                Sampling rate in Hz
            sources : np.ndarray or None (n_sources x n_samples)
                The predicted sources
            unmixing_weights: np.ndarray or None
                Weights of the unmixing matrix
            scores : dict
                Dictonary of source quality scores    


        Returns
        -------
            data (np.ndarray): Pre-prcessed time series data (n_channels x n_samples)
            metadata (dict): TODO    
        
        """


        metadata = {}
        metadata["fsamp_out"] = fsamp

        # Mask bad sources
        n_sources = sources.shape[0]
        source_mask = np.ones(n_sources, dtype=bool)

        if self.steps is not None:
            for step in self.steps:
                
                if isinstance(step, self.RemoveDuplicates):
                    if step.window is not None:
                        t_start = step.window[0] 
                        t_end = step.window[1]
                    else:
                        t_start = 0
                        t_end = sources.shape[1] / fsamp
                    myscores = scores["sil"]
                    local_mask = self._detect_duplicates(
                        spikes,
                        myscores,
                        fsamp,
                        t_start=t_start,
                        t_end=t_end,
                        threshold=step.threshold,
                        max_shift=step.max_shift,
                        tol=step.tolerance,
                        keep_mask=source_mask
                    )
                    source_mask = source_mask & local_mask
                elif isinstance(step, self.BadSourceDetection):
                    if step.quality_metric == "sil":
                        myscores = scores["sil"]
                    elif step.quality_metric == "cov_isi":
                        myscores = scores["cov_isi"]
                    local_mask = self._detect_bad_sources(
                        spikes,
                        myscores,
                        threshold=step.threshold,
                        min_num_spikes=step.min_spikes,
                        keep_mask=source_mask
                    )
                    source_mask = source_mask & local_mask
                elif isinstance(step, self.MaskSources):
                    local_mask = np.ones(n_sources, dtype=bool)
                    local_mask[step.unit_ids] = False
                    source_mask = source_mask & local_mask
                elif isinstance(step, self.ApplyUnmixing):
                    sources, spikes, local_scores = self._apply_unmxing(
                        data=data,
                        fsamp=fsamp,
                        weights=unmixing_weights,
                        ext_factor=step.ext_factor,
                        whiten=step.whiten_data,
                        t_start=step.t_start,
                        t_end=step.t_end,
                    )
                    scores.update(local_scores)
                elif isinstance(step, self.RefineSources):
                    pass
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        
        new_spikes, label_map = self._filter_events(spikes, source_mask)
        new_sources = sources[source_mask, :]
        new_scores = {
            "sil": scores["sil"][source_mask]
        }
        new_weights = unmixing_weights[:, source_mask]

 
        return new_spikes, new_sources, new_scores, new_weights
    


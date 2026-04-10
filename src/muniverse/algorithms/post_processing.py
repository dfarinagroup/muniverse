import warnings
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import welch
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
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

        **Remove Duplicates**: TODO::

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

        **Time window**: Truncate your signal to only consider a selected time window.
        If t_end = -1 the time window ends with the last sample::  

            {
                "step": "time_window",
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
        tail: Literal[-1 ,1] = 1

    class MaskSources(BaseModel):
        step: Literal["mask_sources"]
        unit_id: list[int] = []    
  
    PostProcessStep = Annotated[
        Union[RemoveDuplicates, 
            BadSourceDetection,
            MaskSources,  
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

    def post_process(
            self, 
            events: pd.DataFrame, 
            sources: np.ndarray, # (n_sources x n_samples)
            unmixing_weights: np.ndarray,
            scores: dict,
            fsamp: float = 2048,
    ):
        
        """
        Pre process multi-channel time series data using the 
        specified list of steps.

        Args
        ----
            data (np.ndarray): Raw time series data (n_channels x n_samples)
            fsamp (float): Sampling rate in Hz

        Returns
        -------
            data (np.ndarray): Pre-prcessed time series data (n_channels x n_samples)
            metadata (dict): TODO    
        
        """


        metadata = {}
        metadata["fsamp_out"] = fsamp

        # Mask bad sources
        keep_mask = np.ones(sources.shape[0], dtype=bool)

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
                        events,
                        myscores,
                        fsamp,
                        t_start=t_start,
                        t_end=t_end,
                        threshold=step.threshold,
                        max_shift=step.max_shift,
                        tol=step.tolerance,
                        keep_mask=None
                    )
                    keep_mask *=local_mask
                elif isinstance(step, self.BadSourceDetection):
                    if step.quality_metric == "sil":
                        myscores = scores["sil"]
                    elif step.quality_metric == "cov_isi":
                        myscores = scores["cov_isi"]
                    local_mask = self._detect_bad_sources(
                        events,
                        myscores,
                        threshold=step.threshold,
                        min_num_spikes=step.min_spikes,
                        keep_mask=None
                    )
                    keep_mask *=local_mask
                elif isinstance(step, self.MaskSources):
                    pass
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        
        new_events = self._filter_events(events, keep_mask)
        new_sources = sources[keep_mask, :]
        new_scores = {
            "sil": scores["sil"][keep_mask]
        }
        new_weights = unmixing_weights[:, keep_mask]

 
        return new_events, new_sources, new_scores, new_weights
    


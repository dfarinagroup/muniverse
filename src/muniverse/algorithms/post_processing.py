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
    spike_dict_to_long_df
)    
from ..evaluation.evaluate import (
    pseudo_sil_score
)

class _process_cbss(_BaseCBSS):

    def __init__(
            self, 
            ext_fact = 12, 
            whitening_method = "ZCA", 
            whitening_reg = "auto", 
            spike_detection_exp = 2, 
            spike_detection_min_delay = 0.01, 
            max_spike_shifts = 0.01,
            verbose = False
        ):
        super().__init__(
            ext_fact = ext_fact, 
            whitening_method = whitening_method, 
            whitening_reg = whitening_reg, 
            spike_detection_exp = spike_detection_exp, 
            spike_detection_min_delay = spike_detection_min_delay, 
            verbose = verbose
        )

        self.max_spike_shifts = max_spike_shifts

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
        """Calculate the unmixng weights from data and spikes"""

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

        return new_spikes, sources, scores

    def _optimze_delay(self, X, spikes, fsamp):
        """Helper function"""

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
        quality_metric: Literal["sil", "cov_isi"] = "sil"
        window: tuple[float, float] | None = None
        mode: Literal["max", "min"] = "max"

    class BadSourceDetection(BaseModel):
        step: Literal["bad_source_detection"]
        quality_metric: Literal["sil", "cov_isi"]
        threshold: float = 0.9
        min_spikes: int = 10
        mode: Literal["above", "below"] = "below" 

    class MaskSources(BaseModel):
        step: Literal["mask_sources"]
        unit_ids: list[int] = []   

    class ApplyUnmixing(BaseModel):
        step: Literal["apply_unmixing"]
        rewhiten: bool = True
        ext_factor: int = 12 
        t_start: float = 0
        t_end: float = -1 

    class FitSpikes(BaseModel):
        step: Literal["fit_spikes"]
        ext_factor: int = 12 
        t_start: float = 0
        t_end: float = -1     
                          
    PostProcessStep = Annotated[
        Union[
            RemoveDuplicates, 
            BadSourceDetection,
            MaskSources,
            ApplyUnmixing,
            FitSpikes,
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PostProcessStep)

    def add_step(self, step):
        """ Add an additional post processing step"""
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

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
            ext_factor: int,
            rewhiten: bool = True,
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
            ext_factor : int
                Extension factor 
            rewhiten : bool
                If True, update the whitening matrix given the data
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

        # Extract time window samples
        sample_idx = self._get_win_samples(data, fsamp, t_start, t_end)

        # Apply the given unmixing weights
        cbss = _process_cbss(
            ext_fact=ext_factor
        ) 
        if rewhiten:
            cbss.rewhiten(data[:, sample_idx])
            self.whiten_ = cbss.whiten_
        else:
            cbss.whiten_ = self.whiten_



        if self.unmixing_weights_ is not None:
            if self.unmixing_format_ == "white":
                cbss.unmixing_weights_ = self.unmixing_weights_
            else:
                cbss.unmixing_weights_ = self.whiten_ @ self.unmixing_weights_

            new_spikes, sources, scores = cbss.predict(
                sig = data[:, sample_idx],
                fsamp = fsamp
            )
        else:
            raise ValueError(
                "The unmixing weights are not defined"
            )     
      
        return new_spikes, sources, scores 
    
    def _fit_spikes(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            fsamp: float,
            spikes: pd.DataFrame,
            ext_factor: int,
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
            spikes : pd.DataFrame 
                Learned weights of the unmixing matrix    
            ext_factor : int
                Extension factor 
            t_start : float
                Start time of the considered time window in seconds  
            t_end : float
                End time of the considered time window in seconds                

        Returns
        -------
            spikes : dict 
                Sample indices of motor neuron discharges
            sources : np.ndarray 
                Estimated sources / ica components (n_components x n_samples)
            scores : dict
                Dictonary of source quality scores
        
        """

        # Extract time window samples
        sample_idx = self._get_win_samples(data, fsamp, t_start, t_end)

        # Apply the given unmixing weights
        cbss = _process_cbss(
            ext_fact=ext_factor
        ) 

        if spikes is not None:

            new_spikes, sources, scores = cbss.fit_predict_from_spike_labels(
                sig = data[:, sample_idx],
                spikes=spikes,
                fsamp = fsamp
            )
            self.whiten_ = cbss.whiten_
            self.unmixing_weights_ = cbss.unmixing_weights_
            self.unmixing_format_ = "white"
        else:
            raise ValueError(
                "No spikes are availible for fitting"
            )     
      
        return new_spikes, sources, scores 
    
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
        sample_idx = np.argwhere((t >= t_start) & (t <= t_end)).flatten()

        return sample_idx


    def post_process(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            spikes: pd.DataFrame, 
            fsamp: float,
            scores: dict | None,
            sources: np.ndarray | None = None, # (n_sources x n_samples)
            unmixing_weights: np.ndarray | None = None,
            whitening_matrix: np.ndarray | None = None, 
            unmixing_format: Literal["white", "extended"] = "white"
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
                
                if isinstance(step, self.RemoveDuplicates):
                    if step.window is not None:
                        t_start = step.window[0] 
                        t_end = step.window[1]
                    else:
                        t_start = 0
                        t_end = sources.shape[1] / fsamp
                    myscores = scores["sil"]
                    local_mask = get_duplicates_mask(
                        spikes=spikes,
                        scores=myscores,
                        fsamp=fsamp,
                        mode=step.mode,
                        t_start=t_start,
                        t_end=t_end,
                        duplicate_theshold=step.threshold,
                        max_shift=step.max_shift,
                        tol=step.tolerance,
                        mask=source_mask
                    )
                    source_mask = source_mask & local_mask
                elif isinstance(step, self.BadSourceDetection):
                    if step.quality_metric == "sil":
                        myscores = scores["sil"]
                    elif step.quality_metric == "cov_isi":
                        myscores = scores["cov_isi"]
                    local_mask = get_bad_source_mask(
                        spikes=spikes,
                        score=myscores,
                        threshold=step.threshold,
                        mode=step.mode,
                        min_num_spikes=step.min_spikes
                    )
                    source_mask = source_mask & local_mask
                elif isinstance(step, self.MaskSources):
                    local_mask = np.ones(n_sources, dtype=bool)
                    local_mask[step.unit_ids] = False
                    source_mask = source_mask & local_mask
                elif isinstance(step, self.ApplyUnmixing):
                    spikes, sources, local_scores = self._apply_unmxing(
                        data=data,
                        fsamp=fsamp,
                        ext_factor=step.ext_factor,
                        rewhiten=step.rewhiten,
                        t_start=step.t_start,
                        t_end=step.t_end,
                    )
                    scores.update(local_scores)
                elif isinstance(step, self.FitSpikes):
                    spikes, sources, local_scores = self._fit_spikes(
                        data=data,
                        fsamp=fsamp,
                        spikes=spikes,
                        ext_factor=step.ext_factor,
                        t_start=step.t_start,
                        t_end=step.t_end,
                    )
                    scores.update(local_scores)    
                # elif isinstance(step, self.RefineSources):
                #     pass
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        
        new_spikes, label_map = self._filter_events(spikes, source_mask)
        new_sources = sources[source_mask, :]
        new_scores = {
            "sil": scores["sil"][source_mask]
        }
        # new_weights = unmixing_weights[:, source_mask]

 
        return new_spikes, new_sources, new_scores
    



import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from pydantic import BaseModel, TypeAdapter, Field
from typing import Literal, List, Union, Annotated
from .core import (
    bandpass_signals, 
    notch_signals, 
    highpass_signals, 
    lowpass_signals,
    find_outliers
)


class pre_processing:
    """
    Class to preprocess HD-EMG data.

    Parameters
    ----------
    steps : list of dict
        List of preprocessing steps. Each step is a dictionary describing
        the processing operation.

        Supported step types are:
        "bandpass", "highpass", "lowpass", "notch",
        "bad_channel_detection", "mask_channels", "downsample":

        **Bandpass filter**: Bandpass filter time series data using a digital
        infinite impulse response filter ("butter") or finite impulse response
        filter ("firwin2")::

            {
                "step": "bandpass",
                "high_pass": float,
                "low_pass": float,
                "method": "butter" | "firwin2",
                "order": int,      # required if method == "butter"
                "numtabs": int,    # required if method == "firwin2"
            }

        **Highpass filter**: Highpass filter time series data using a digital
        infinite impulse response filter ("butter") or finite impulse response
        filter ("firwin2")::

            {
                "step": "highpass",
                "high_pass": float,
                "method": "butter" | "firwin2",
                "order": int, # required if method == "butter"
                "numtabs": int, # required if method == "firwin2"
            }

        **Lowpass filter**: Lowpass filter time series data using a digital
        infinite impulse response filter ("butter") or finite impulse response
        filter ("firwin2")::

            {
                "step": "lowpass",
                "low_pass": float,
                "method": "butter" | "firwin2",
                "order": int, # required if method == "butter"
                "numtabs": int, # required if method == "firwin2"
            }

        **Notch filter**:: Apply a digital notch (stop band) filter using either a
        infinite impulse response filter ("butter"), a finite impulse response 
        filter ("iirnotch") or performing filtering in the frequency domain 
        ("fft_nulling" and "fft_interpolation"). For "fft_nulling" the spectrum in
        the specified frequency band is set to zero, for "fft_interpolation" the 
        spectral amplitude is interpolated through by the neighbourhood. Time series
        data is recovered through an inverse fft.

            {
                "step": "notch",
                "freqs": list[float],
                "method": "butter" | "iirnotch" | "fft_nulling" | "fft_interpolation",
                "order": int,   # if "butter"
                "dfreq": float  # if "iirnotch", "fft_nulling" or "fft_interpolation"
            }

        **Bad Channel Detection**: Automatically detect bad channels based on some 
        metric ("std" or "rms") computed in a given time window (given in seconds). 
        If method is "zscore" the score distribution is normalized (zero mean, 
        unit standard deviation). All scores are compared to a "threshold_value". 
        If tail=1 all values above the threshold are rejected, if tail=-1 all values 
        below the theshold are rejected. For tail=0 all the absolute value of the score
        is computed and all values above the threshold are rejected (only availible 
        if "method" == "zscore")::  

            {
                "step": "bad_channel_detection",
                "metric": Literal["std", "rms", "medfreq", "medpower"],
                "window": (t0, t1) | None, # Given in seconds, if None consider the full data
                "method": "zscore" | "threshold",
                "threshold_value": float,
                "tail": -1 | 0 | 1
            } 

        **Mask Channels**: Mask all channels given in "channel_list" to be excluded in the following. 
        Can be either used to reject known bad channels or limit the analysis to a subset of your data::  

            {
                "step": "mask_channels",
                "channel_list": list[int]
            }

        **Downsample**: Reduces the sampling frequency by the specified value::  

            {
                "step": "downsample",
                "factor": int 
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
    Pre process HD-EMG data using a bandpass and a notch filter.
    >>> model = pre_processing(steps = [
    ...     {
    ...         "step": "bandpass",
    ...         "high_pass": 20,
    ...         "low_pass": 500,
    ...         "method": "butter",
    ...         "order": 2
    ...     },
    ...     {
    ...         "step": "notch",
    ...         "freqs": [50, 100, 150],
    ...         "method": "butter",
    ...         "order": 2
    ...     },
    ... ])
    >>> preprocessed_data, metadata = model.pre_process(data=emg_data, fsamp=2048)                     

    """

    def __init__(
            self, 
            steps: list[dict] = [
                {  
                    "step": "bandpass",
                    "high_pass": 20,
                    "low_pass": 500,
                    "method": "butter",
                    "order": 2,
                },
                {
                    "step": "notch",
                    "freqs": [50, 100, 150],
                    "method": "butter",
                    "order": 2
                }    
            ]          
    ):

        #self.pre_process_steps = pre_process_steps
        self.steps = [
            self._adapter.validate_python(step)
            for step in steps
        ]

    class Bandpass(BaseModel):
        step: Literal["bandpass"]
        high_pass: float = 20
        low_pass: float = 500
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101

    class Highpass(BaseModel):
        step: Literal["highpass"]
        high_pass: float = 20
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101 

    class Lowpass(BaseModel):
        step: Literal["lowpass"]
        low_pass: float = 500
        method: Literal["butter", "firwin2"] = "butter"
        order: int = 2    
        numtabs: int = 101       

    class Notch(BaseModel):
        step: Literal["notch"]
        freqs: List[float] = [50, 100, 150]
        method: Literal[
            "butter", "iirnotch", "fft_nulling", "fft_interpolation"
            ] = "butter" 
        order: int = 2
        dfreq: float = 1

    class BadChannelDetection(BaseModel):
        step: Literal["bad_channel_detection"]
        metric: Literal["std", "rms", "medfreq", "medpower", "cumpower"]
        method: Literal["zscore", "threshold"] = "zscore"
        threshold_value: float = 3
        max_iter: int | None = 3
        tail: Literal[-1, 0, 1] = 0 
        window: tuple[float, float] | None = None
        bandwidth: tuple[float, float] | None = None   

    class MaskChannels(BaseModel):
        step: Literal["mask_channels"]
        channel_list: list[int] = []    

    class Downsample(BaseModel):
        step: Literal["downsample"]
        factor: int

    class TimeWindow(BaseModel):
        step: Literal["time_window"]
        t_start: float = 0
        t_end: float = -1    

    PreprocessStep = Annotated[
        Union[Bandpass, 
            Lowpass,
            Highpass,  
            Notch, 
            BadChannelDetection, 
            MaskChannels, 
            Downsample,
            TimeWindow
        ],
        Field(discriminator="step")
    ]  

    _adapter = TypeAdapter(PreprocessStep)

    def add_step(self, step):
        """ Add an additional post processing step"""
        
        self.steps.append(
            self._adapter.validate_python(step)
        )

    def _get_scores(
            self, 
            data: np.ndarray, # (n_channels, n_samples) 
            metric: Literal[
                "rms", "std", "medfreq", "medpower", "cumpower"
            ], 
            fsamp: float | None = 2048, 
            bw: tuple | None = (20, 500)
    ):
        """
        Calculate channel specific scores

        Args
        ----
            data : np.ndarray (n_channels, n_samples)
                time series data 
            metric : {"rms", "std", "medfreq", "medpower", "cumpower"}
                Specify the computed metric. Can be the root-mean-square ("rms"),
                the standard deviation ("std"), the median frequency ("medfreq"),
                the median power ("medpower") or the cumulative power ("cumpower")
            fsamp : float
                If your metric is "medfreq", "medpower", "cumpower" you need
                to specify the sampling rate in Hz
            bw : tuple of float
                If your metric is "medfreq", "medpower", "cumpower" you need
                to specify the considered bandwidth in Hz     

        Returns
        -------
            score : np.ndarray
                Array of score values (n_channels, )
        
        """

        METRICS = ["rms", "std", "medfreq", "medpower", "cumpower"]
                    
        if metric == "rms":
            score = np.mean(data**2, axis=1)**0.5
        elif metric == "std":
            score = np.std(data, axis=1)
        elif metric == "medfreq":
            psd, freqs = welch(data, fs=fsamp, nperseg=fsamp, noverlap=fsamp/2)
            idx = np.where((freqs > bw[0]) & (freqs < bw[1]))[0]
            cumulative = np.cumsum(psd[:, idx], axis=1)
            total = cumulative[:, -1][:, None]
            med_idx = np.argmax(cumulative >= total / 2, axis=1)
            score = freqs[med_idx]
        elif metric == "medpower":
            psd, freqs = welch(data, fs=fsamp, nperseg=fsamp, noverlap=fsamp/2)
            idx = np.where((freqs > bw[0]) & (freqs < bw[1]))[0]
            score = np.median(psd[:,idx], axis=1)
        elif metric == "cumpower":
            psd, freqs = welch(data, fs=fsamp, nperseg=fsamp, noverlap=fsamp/2)
            idx = np.where((freqs > bw[0]) & (freqs < bw[1]))[0]
            score = np.sum(psd[:,idx], axis=1)
        else:
            raise ValueError(
                f"Invalid metric {metric}"
                f"Must be one of {METRICS}"
            )
        
        return score

    def _get_bad_channels(
            self, 
            score: np.ndarray, # (n_channels, ) 
            mask: np.ndarray[bool], # (n_channels, )  
            method: Literal["zscore", "threshold"], 
            threshold_value: float, 
            max_iter: int | None = 3, 
            tail: Literal[-1, 0, 1] = 0
    ):
        """
        Automatically detect bad channels based on the scores of
        channel-specific metrics and a given threshold value. 
        Either using a fixed threshold  (method = "threshold") or 
        z-score normalized scores (method = "zscore").

        Args
        ----
            score : np.ndarray
                Channel-specific scores
            mask : np.ndarry
                Boolean mask of considered channels (True: not used, False: used)
            method : {"zscore", "threshold"}
                Method used for bad channel detection. If "zscore" 
                scores are z-score normalized prior to thresholding.
            threshold_value : float
                Treshold for bad channel detection
            tail : {-1, 0, 1}
                If 1 flag all values above the threshold;
                If -1 flag all values below the threshold;
                If 0 (only availible if method = "zscore"), flag all 
                channels with an absolute score above the threshold                    

        Returns
        -------
            mask : np.ndarray of bool
                Boolean mask (True: bad channel, False: good channel)
        
        """

        if method == "zscore":
            mask = find_outliers(
                score, threshold_value, max_iter=max_iter, tail=tail, mask=mask
            )
        elif method == "threshold":
            if tail == 1:
                mask = score > threshold_value
            elif tail == -1:
                mask = score < threshold_value
            else:
                raise ValueError(
                    f"For method *{method}* tail must be -1 or 1."
                )
        else:
            raise ValueError(
                "Invalid bad channel detection method"
                "Must be one of *zscore* or *threshold*"
            )

        return mask             

    def pre_process(
            self, 
            data: np.ndarray, # (n_channels x n_samples)
            fsamp: float = 2048,
    ):
        
        """
        Pre process multi-channel time EMG data using the 
        specified list of steps.

        Args
        ----
            data : np.ndarray (n_channels x n_samples)
                Raw time series data 
            fsamp : float 
                Sampling rate in Hz

        Returns
        -------
            data : np.ndarray (n_channels x n_samples)
                Pre-prcessed time series data 
            metadata : dict
                Dictonary of process metadata 
                - fsamp (float): Sampling rate in Hz
                - ch_mask (np.ndarray): Boolean channel selection mask
                - sample_mask (np.ndarray): Boolean sample selection mask
                - ch_status (pd.DataFrame): Channel status 
            steps : list
                List of the applied processing steps       
        
        """

        # Initalize the pocess metadata
        fsamp_new = fsamp
        ch_mask = np.ones(data.shape[0], dtype=bool)
        sample_mask = np.ones(data.shape[1], dtype=bool)
        ch_status = pd.DataFrame({
            "name": [f"Ch{i:03d}" for i in range(1, data.shape[0] + 1)],
            "status": ["on"] * data.shape[0],
            "status_description": ["n/a"] * data.shape[0],
        })

        # Loop over all steps
        if self.steps is not None:
            for step in self.steps:
                
                if isinstance(step, self.Bandpass):
                    data = bandpass_signals(
                        data,
                        fsamp_new,
                        high_pass=step.high_pass,
                        low_pass=step.low_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.Notch):
                    data = notch_signals(
                        data,
                        fsamp_new,
                        freqs=step.freqs,
                        method=step.method,
                        order=step.order,
                        dfreq=step.dfreq,
                    )  
                elif isinstance(step, self.Highpass):
                    data = highpass_signals(
                        data,
                        fsamp_new,
                        high_pass=step.high_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.Lowpass):
                    data = lowpass_signals(
                        data,
                        fsamp_new,
                        low_pass=step.low_pass,
                        method=step.method,
                        order=step.order,
                        numtabs=step.numtabs,
                    )
                elif isinstance(step, self.MaskChannels):
                    local_mask = np.ones(data.shape[0], dtype=bool)
                    local_mask[step.channel_list] = False
                    ch_mask = ch_mask & local_mask
                    ch_status.loc[~local_mask, "status"] = "off"
                    ch_status.loc[
                        ~local_mask, "status_description"
                    ] = step.model_dump().__str__()
                elif isinstance(step, self.BadChannelDetection):
                    if step.window is not None:
                        idx0 = int(step.window[0] / fsamp_new)
                        idx1 = int(step.window[1] / fsamp_new)
                    else:
                        idx0 = 0
                        idx1 = data.shape[1]
                    scores = self._get_scores(data[:, idx0:idx1], step.metric)
                    bad_mask = self._get_bad_channels(
                        scores,
                        ~ch_mask,
                        method=step.method,
                        threshold_value=step.threshold_value,
                        max_iter=step.max_iter,
                        tail=step.tail
                    )
                    ch_mask = ch_mask & ~bad_mask
                    ch_status.loc[bad_mask, "status"] = "off"
                    ch_status.loc[
                        bad_mask, "status_description"
                    ] = step.model_dump().__str__()
                elif isinstance(step, self.Downsample):
                    data = data[:, ::step.factor]
                    sample_mask = sample_mask[::step.factor]
                    fsamp_new = fsamp_new / step.factor
                elif isinstance(step, self.TimeWindow):
                    n_samples = data.shape[1]
                    t = np.linspace(0, (n_samples-1) / fsamp_new, n_samples)
                    sample_mask = np.argwhere((t >= step.t_start) & (t <= step.t_end)).flatten()
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        # Package process metadata in a dictonary        
        metadata = {}
        metadata["ch_mask"] = ch_mask
        metadata["sample_mask"] = sample_mask
        metadata["ch_status"] = ch_status
        metadata["fsamp"] = fsamp_new

        # Package the applied processing steps
        steps = [step.model_dump() for step in self.steps]

        return data, metadata, steps


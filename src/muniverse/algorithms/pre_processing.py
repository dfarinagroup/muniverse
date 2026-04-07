import warnings
import numpy as np
from scipy.stats import zscore
from scipy.signal import welch
from .core import bandpass_signals, notch_signals, find_outliers

class pre_processing:
    """
    Class to preproces HD-EMG data

    """

    def __init__(self, 
                 fsamp: float = 2048, 
                 pre_process_steps: list | None = [
                    {
                        "step": "bad_channel_detection",
                        "metric": "std",
                        "method": "zscore",
                        "value": 3,
                        "tail": 0,
                    },
                    {
                        "step": "bad_channel_detection",
                        "metric": "rms",
                        "method": "threshold",
                        "value": 1e-6, 
                        "tail": -1,
                    },
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


        self.fsamp = fsamp
        self.pre_process_steps = pre_process_steps


    def add_step(self, step):
        
        self.pre_process_steps.append(step)

    def _get_scores(self, data, metric, bw=[20, 500]):

        METRICS = ["rms", "std", "medfreq", "medpower"]
                    
        if metric == "rms":
            score = np.mean(data**2, axis=1)**0.5
        elif metric == "std":
            score = np.std(data, axis=1)
        elif metric == "medfreq":
            freqs, psd = welch(data, fs=self.fsamp, nperseg=self.fsamp/2)
            cumulative = np.cumsum(psd)
            total = cumulative[-1]
            idx = np.where(cumulative >= total / 2)[0][0]
            score = freqs[idx]
        elif metric == "medpower":
            freqs, psd = welch(data, fs=self.fsamp, nperseg=self.fsamp/2)
            total = cumulative[-1]
            score = np.median(psd, axis=1)
        else:
            raise ValueError(
                f"Invalid metric {metric}"
                f"Must be one of {METRICS}"
            )
        
        return score
    
    def _find_outliers(self, x, threshold=3, max_iter=3, tail=0):
        """
        Detect ouliers by comparing the z-score of variable x against
        some threshold. This is repeaded until there are no outliers or
        the maximum number of iterations is reached. 

        Args:
            x (np.array): Variable to test for outliers
            threshold (float): Threshold for outlier detection
            max_iter (int): Maximum number of iterations
            tail {-1,0,1}: Specify weather to serach for outliers   
                on both ends (0), just on the positive (1) or just 
                the negative side (-1).

        Return:
            bad_idx (np.array): List of bad channels (integer index) 
            
        """

        mask = np.zeros(len(x), dtype=bool)

        iter = 0
        while iter < max_iter:
            xm = np.ma.masked_array(x, mask=mask)
            if tail == 1:
                idx = zscore(xm) > threshold
            elif tail == -1:
                idx = -zscore(xm) > threshold
            else:
                idx = np.abs(zscore(xm)) > threshold 
            mask += idx
            if not np.any(idx):
                break
            else:
                iter = iter + 1  

        return mask 

    def _get_bad_channels(self, score, mask, method, threshold, max_iter=3, tail=0):

        if method == "zscore":
            mask = self._find_outliers(score, threshold, max_iter=max_iter, tail=tail)
        elif method == "threshold":
            if tail == 1:
                mask = score > threshold
            elif tail == -1:
                mask = score < threshold
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

    def pre_process(self, data: np.ndarray):


        meta = {}

        # Mask bad channels
        mask = np.zeros(data.shape[0], dtype=bool)

        if self.pre_process_steps is not None:
            for cfg in self.pre_process_steps:
                
                if cfg["step"] == "bandpass":
                    data = bandpass_signals(
                        data,
                        self.fsamp,
                        high_pass=cfg.get("high_pass", 20),
                        low_pass=cfg.get("low_pass", 500),
                        ftype=cfg.get("method", "butter"),
                        order=cfg.get("order", 2),
                        numtabs=cfg.get("numtabs", 101),
                    )
                elif cfg["step"] == "notch":
                    data = notch_signals(
                        data,
                        self.fsamp,
                        nfreq=cfg.get("freqs", [50, 100, 150]),
                        ftype=cfg.get("method", "butter"),
                        order=cfg.get("order", 2),
                        dfreq=cfg.get("dfreq", 1),
                        n_harmonics=1
                    )  
                elif cfg["step"] == "highpass":
                    pass
                elif cfg["step"] == "lowpass":
                    pass
                elif cfg["step"] == "mask_channels":
                    local_mask = np.zeros(data.shape[0], dtype=bool)
                    local_mask[cfg["channel_list"]] = True
                    mask += local_mask
                elif cfg["step"] == "bad_channel_detection":
                    scores = self._get_scores(data, cfg["metric"])
                    local_mask = self._get_bad_channels(
                        scores,
                        mask,
                        method=cfg.get("method"),
                        threshold=cfg.get("value"),
                        max_iter=cfg.get("max_iter", 3),
                        tail=cfg.get("tail", 0)
                    )
                    mask += local_mask
                elif cfg["step"] == "downsample":
                    data = data[:, ::cfg]
                    meta["fsamp_out"] = self.fsamp / cfg
                else:
                    raise ValueError(
                        "Invalid step type"
                    )
        
        selected_idx = np.where(mask == False)[0]    
            
        data = data[selected_idx, :]
        meta["mask"] = mask

 
        return data, meta


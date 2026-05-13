import numpy as np
from typing import List, Literal, Optional
from scipy.stats import skew
from scipy.signal.windows import tukey
from .core import (
    extension, 
    est_spike_times, 
    spike_dict_to_long_df
)
from .cbss import _BaseCBSS


class UpperBoundCBSS(_BaseCBSS):
    """
    Class for computing an upper bound of convolutive blind source
    separation (CBSS) based motor neuron identification making use
    of known ground-truth motor unit response waveforms.

    - Transform the convolutive mixture into an instantaneous mixture 
    by adding R delayed copies of the input signal
    - Apply a whitening transformation to the extended signals to 
    obtain data with unit variance
    - Obtain for each motor unit the optimal unmixing weights it's 
    impulse response waveform 
    - Extract the motor unit spikes using peak-detection and 
    spike clustering

    Properties
    ----------
        ext_fact : int , default 12
            Extension factor

        whitening_method : {"ZCA", "PCA", "Cholesky"}, default "ZCA" 
            Method used for whitening

        whitening_backend : {"ed", "svd"}, default "ed" 
            Method used to calculate eigenvalues and eigenvectors. Can be
            either based on singular value decomposition ("svd") or an
            eigendecomposition ("ed"). Only needed if method is "ZCA" or "PCA". 

        whitening_regularization : {"auto", float, None}, default "auto" 
            Adds a small value to the eigenvalues for regularization. 
            If "auto", the mean of the second half of the eigenvalues is used.

        spike_detection_exp : float , default 2
            Exponent of asymetric power law applied to the extracted sources
            before spike detection

        spike_detection_min_delay : float , default 0.01
            Minimum distance between two detected spikes in seconds  

        win_alpha : float , default 0 
            Shape parameter of a Tukey window (representing the fraction of the window 
            inside the cosine tapered region) that is applied to the MUAPs
            to minimize edge effects. If zero, the Tukey window is equivalent to a 
            rectangular window. If one, the Tukey window is equivalent to a Hann window. 

        verbose : float , default True
            If True, print progress. 

    Attributes
    ----------
        unmixing_weights_ : np.ndarray (n_features, n_components)
            The learned unmixing weights

        whiten_ : np.ndarray (n_features, n_features)
            Whitening matrix   

        unwhiten_ : np.ndarray (n_features, n_features)
            Inverse of the whitening matrix  

        expected_amplitudes_ : np.ndarray 
            For each motor unit impulse response and each delay the expected 
            spike amplitude. The algorithm selects for each motor unit
            the maximum value.  

    References
    ----------
    .. [1] Klotz and Rohlen, "Revisiting convolutive blind source separation 
           for identifying spiking motor neuron activity: from theory to 
           practice", Journal of Neural Engineering, 2025 
    .. [2] Mamidanna et et al., "MUniverse: A Simulation and Benchmarking 
           Suite for Motor Unit Decomposition", The Thirty-ninth Annual 
           Conference on Neural Information Processing Systems 
           Datasets and Benchmarks Track, 2025                    


    Example
    -------

    Init UpperBoundCBSS class using the default parameters and run decomposition.
    >>> model = UpperBoundCBSS() 
    >>> spikes, sources, scores = model.fit_predict(sig=emg_data, muaps=muaps, fsamp=2048)        

    """

    def __init__(
            self,
            ext_fact: int = 12,
            whitening_method: Literal["ZCA", "PCA", "Cholesky"] = "ZCA",
            whitening_backend: Literal["ed", "svd"] = "ed",
            whitening_reg: str | float | None = "auto",
            spike_detection_exp: float = 2,
            spike_detection_min_delay: float = 0.01,
            win_alpha: float = 0,
            verbose: bool = False,
            config: dict | None = None
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

        self.win_alpha = win_alpha

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        self._params = set(self.__dict__.keys()) - {"_params"}

        # Set all parameters from the config dict
        for key, value in config_dict.items():
            if key in self._params:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

        self._attributes = set([
            "unmixing_weights_", "whiten_", 
            "unwhiten_", "expected_amplitudes_"
        ])        

    def fit_predict(
            self, 
            sig: np.ndarray, # (n_channels, n_samples) 
            muaps: np.ndarray, # (n_mu, n_channels, n_samples)
            fsamp: float
    ):
        """
        Estimate the spike response of motor neurons given the
        motor unit impulse response waveforms (MUAPs)

        Args
        ----
            sig : np.ndarray 
                Input data (n_channels, n_samples)
            muaps : np.ndarray 
                Impulse response waveforms (n_units, n_channels, n_samples)
            fsamp : float
                 Sampling rate in Hz

        Returns
        -------
            spikes : pd.DataFrame 
                Spike table (columns: onset, duration, sample, unit_id, description)
            sources : np.ndarray 
                Estimated sources (n_components, n_samples)
            scores : dict of np.ndarray 
                Source trustworthiness scores ("sil" and "cov_isi") 
        
        """

        # Extend signals and subtract the mean
        ext_sig = self._extension(sig)

        # Whiten the extended signals
        white_sig = self._whitening(ext_sig)

        # Init sources
        n_mu = muaps.shape[0]
        sources = np.zeros((n_mu, white_sig.shape[1]))

        # Init spikes and scores
        spikes = {i: [] for i in range(sources.shape[0])}
        scores = {
            "sil": np.zeros(n_mu),
            "cov_isi": np.zeros(n_mu),
        }
        # Initialize unmixing weights
        self.unmixing_weights_ = np.zeros((white_sig.shape[0], n_mu))
        self.expected_amplitudes_ = np.zeros((n_mu, muaps.shape[2]))


        # Loop over each MU
        for i in np.arange(n_mu):
            # Get the optimal MU filter
            w = self._get_optimal_unmixing_weights(muaps[i, :, :], i)
            # Estimate source
            sources[i, :] = w.T @ white_sig
            # Make sure the peaks are in positive direction
            sign = np.sign(skew(sources[i, :]))
            sources[i, :] = sign * sources[i, :]
            spikes[i], scores["sil"][i] = est_spike_times(
                source = sources[i, :], 
                fsamp = fsamp, 
                a = self.spike_detection_exp,
                min_delay = self.spike_detection_min_delay
            )
            scores["cov_isi"][i] = self._calc_cov_isi(spikes[i], fsamp)
            # Store the unmixing weights
            self.unmixing_weights_[:, i] = w * sign

        # Convert dict of spikes to long-formated spike table 
        spikes = spike_dict_to_long_df(spikes, fsamp=fsamp)    

        return spikes, sources, scores

    def _get_optimal_unmixing_weights(
        self, 
        muap: np.ndarray,
        i: int
    ) -> np.ndarray:
        """
        Get the optimal unmixing weights from the ground truth motor unit
        impulse response waveforms (MUAP). Therefore, the MUAP is extended 
        and whitened. The optimal unmixing weight corresponds to the column 
        of the extended and whitened MUAP that has the highest norm.

        Args
        ----
            muap : np.ndarray 
                Impulse response waveform (n_channels x n_samples)
            i : int
                Current iteration

        Returns
        -------
            w : np.ndarray 
                Optimal unmixing weights 

        """

        # Apply tukey window
        win = tukey(M=muap.shape[1], alpha=self.win_alpha).reshape(-1,1)
        muap = win.T * muap

        # Extend the MUAP
        ext_muap = extension(muap, self.ext_fact)
        # ext_muap -= ext_mean

        # Whiten the MUAP
        white_muap = self.whiten_ @ ext_muap

        # Find the column with the largest L2 norm and return it as MUAP filter
        col_norms = np.linalg.norm(white_muap, axis=0)
        col_norms[: self.ext_fact] = 0
        w = white_muap[:, np.argmax(col_norms)]

        # Normalize w
        w = w / np.linalg.norm(w)

        self.expected_amplitudes_[i, :] = col_norms

        return w
    
    
# TODO Move the following files somewhere else     

def process_neuromotion_muaps(muap_cache, simulation_config):
    """
    Load and prepare MUAPs for decomposition for a given simulated recording.
    I.e., pick MUAPs for target angle --> select subset of electrodes used during simulation

    Args:
        muap_cache: path to MUAP cache file
        simulation_config: simulation config dict

    Returns:
        processed_muaps: Processed MUAPs ready for decomposition
    """    
    # Extract configuration from the simulation config
    config = simulation_config.get('InputData', {}).get('Configuration', {})
    movement_config = config.get('MovementConfiguration', {})
    movement_dof = movement_config.get('MovementDOF')
    
    # Generate angle labels
    if movement_dof == "Flexion-Extension":
        min_angle, max_angle = -65, 65
    elif movement_dof == "Radial-Ulnar-deviation":
        min_angle, max_angle = -10, 25

    constant_angle = movement_config.get("MovementProfileParameters", {}).get('TargetAngle')
    muap_dof_samples = muap_cache.shape[1]
    angle_labels = np.linspace(min_angle, max_angle, muap_dof_samples).astype(int)

    # Find the index of the angle in the MUAP cache
    angle_idx = np.argmin(np.abs(angle_labels - constant_angle))
    muaps = muap_cache[:, angle_idx, :, :, :]

    # Reshape MUAPs from (n_mu, n_rows, n_cols, n_samples) to (n_mu, n_channels, n_samples)
    n_mu, n_rows, n_cols, n_samples = muaps.shape
    
    # Check if we need to use subset of electrodes based on simulation config
    selected_indices = None
    electrode_config = config.get('RecordingConfiguration', {}).get('ElectrodeConfiguration', {})
    desired_n_cols = electrode_config.get('DesiredNCols')
    
    if desired_n_cols < n_cols:
        # Biomime grid wraps around -- use modulo to handle wrapping
        # Calculate how many columns to take on each side of the center column
        # TODO: Move the center column from OutputData.Metadata to ElectrodeConfiguration
        center_column = simulation_config['OutputData']['Metadata'].get('CenterColumn')
        half_width = desired_n_cols // 2
        selected_columns = [(center_column - half_width + i) % n_cols for i in range(desired_n_cols)]
        selected_indices = []
        for col in selected_columns:
            selected_indices.extend([col * n_rows + row for row in range(n_rows)])
    
    # Reshape the MUAPs
    processed_muaps = muaps.reshape(n_mu, n_rows * n_cols, n_samples)
    if selected_indices is not None:
        processed_muaps = processed_muaps[:, selected_indices, :]
        print(f"Extracted MUAPs for angle {constant_angle}, and subset of {len(selected_indices)} electrodes")
    else:
        print(f"Extracted MUAPs for angle {constant_angle}, using all {n_rows * n_cols} electrodes")
    
    return processed_muaps


def process_hybrid_tibialis_muaps(muap_cache, subject_config):
    """
    Load and prepare MUAPs for decomposition for a given hybrid tibialis recording.
    """
    return muap_cache[subject_config['simulation_info']['selected_indices']]
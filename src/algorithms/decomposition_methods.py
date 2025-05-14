import numpy as np
from .decomposition_routines import *
from .pre_processing import *
import sys
import os


class upper_bound:
    '''
    Class for computing an upper bound of convolutive blind source 
    separation (cBSS) based motor neuron indedification making use 
    of a known ground-truth.
    '''

    def __init__(self, config=None, **kwargs):
        # Default parameters
        self.ext_fact = 12
        self.whitening_method = 'ZCA'
        self.whitening_reg  = 'auto'
        self.cluster_method  = 'kmeans'
        # Added by DH 
        self.sil_th = 0.9
        self.min_num_spikes = 10

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        # Merge with directly passed keyword arguments (overwrites config)
        params = {**config_dict, **kwargs}

        valid_keys = self.__dict__.keys()

        # Assign all parameters as attributes
        for key, value in params.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

    def set_param(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")    
                
    def load_muaps(self, simulation_config_path, muap_cache_file):
        """
        Load and prepare MUAPs for decomposition using a single simulation configuration file.

        Args:
            simulation_config_path: Path to simulation configuration JSON file
            muap_cache_file: Path to MUAP cache file

        Returns:
            muaps_reshaped: Reshaped MUAPs ready for decomposition
            fsamp: Sampling frequency from config
            angle: Angle used for MUAP selection
        """
        # Load simulation configuration file
        import json
        
        with open(simulation_config_path, 'r') as f:
            simulation_config = json.load(f)
        
        # Extract configuration from the simulation config
        config = simulation_config.get('InputData', {}).get('Configuration', {})
        if not config:
            # Fallback: if it's already the direct configuration without the outer structure
            config = simulation_config.get('Configuration', simulation_config)
        
        # Get sampling frequency
        fsamp = config.get('RecordingConfiguration', {}).get('SamplingFrequency')
        if not fsamp:
            raise ValueError("Could not find sampling frequency in simulation config")
        
        # Load MUAPs from cache
        if muap_cache_file is not None:
            print(f"Loading MUAPs from cache: {muap_cache_file}")
            muaps_full = np.load(muap_cache_file, allow_pickle=True)
        else:
            raise ValueError("MUAP cache file is required for decomposition")

        # Extract movement information
        movement_config = config.get('MovementConfiguration', {})
        movement_dof = movement_config.get('MovementDOF')
        
        # Generate angle labels
        if movement_dof == "Flexion-Extension":
            min_angle, max_angle = -65, 65
        elif movement_dof == "Radial-Ulnar-deviation":
            min_angle, max_angle = -10, 25
        else:
            min_angle, max_angle = -65, 65  # Default to Flexion-Extension range
            print(f"Warning: Unknown movement DOF '{movement_dof}'. Using default angle range.")

        constant_angle = movement_config.get("MovementProfileParameters", {}).get('TargetAngle')

        muap_dof_samples = muaps_full.shape[1]
        angle_labels = np.linspace(min_angle, max_angle, muap_dof_samples).astype(int)

        # Find the index of the angle in the MUAP library
        angle_idx = np.argmin(np.abs(angle_labels - constant_angle))
        muaps = muaps_full[:, angle_idx, :, :, :]

        # Reshape MUAPs from (n_mu, n_rows, n_cols, n_samples) to (n_mu, n_channels, n_samples)
        n_mu, n_rows, n_cols, n_samples = muaps.shape
        
        # Check if we need to use subset of electrodes based on simulation config
        selected_indices = None
        
        # Extract electrode array info from config
        electrode_config = config.get('RecordingConfiguration', {}).get('ElectrodeConfiguration', {})
        desired_n_cols = electrode_config.get('DesiredNCols')
        
        # Get selected columns from simulation metadata if available
        if 'OutputData' in simulation_config and 'Metadata' in simulation_config['OutputData']:
            center_column = simulation_config['OutputData']['Metadata'].get('CenterColumn')
            
            # If center column is specified and desired columns is less than total columns
            if center_column is not None and desired_n_cols and desired_n_cols < n_cols:
                # Calculate how many columns to take on each side of the center column
                half_width = desired_n_cols // 2
                
                # Use the same wrapping logic as in run_neuromotion.py
                # Biomime grid wraps around -- use modulo to handle wrapping
                selected_columns = [(center_column - half_width + i) % n_cols for i in range(desired_n_cols)]
                
                # Generate selected indices based on columns
                selected_indices = []
                for col in selected_columns:
                    selected_indices.extend([col * n_rows + row for row in range(n_rows)])
                
                print(f"Using {len(selected_indices)} electrodes (columns: {selected_columns})")
        
        # Reshape the MUAPs
        if selected_indices:
            # First reshape to flattened form
            muaps_flat = muaps.reshape(n_mu, n_rows * n_cols, n_samples)
            # Then select only the required electrodes
            muaps_reshaped = muaps_flat[:, selected_indices, :]
            print(f"Selected {len(selected_indices)} electrodes from original {n_rows * n_cols}")
        else:
            # Use all electrodes
            muaps_reshaped = muaps.reshape(n_mu, n_rows * n_cols, n_samples)
            print(f"Using all {n_rows * n_cols} electrodes")
        
        return muaps_reshaped, fsamp, constant_angle

    def decompose(self, sig, muaps, fsamp):
        """
        Estimate the spike response of motor neurons given the 
        motor unit action potentials (MUAPs)

        Args:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            muaps (ndarray): MUAPs (mu_index x n_channels x duration)
            fsamp (float): Sampling rate of the data (unit: Hz)

        Returns:
            sources (np.ndarray): Estimated sources (n_mu x n_samples)
            spikes (dict): Spiking instances of the motor neurons
            sil (np.ndarray): Source quality metric
            mu_filters (np.ndarray): Motor unit filters
        """

        n_mu = muaps.shape[0]
        sources = np.zeros((n_mu, sig.shape[1]))
        spikes = {i: [] for i in range(n_mu)}
        sil = np.zeros(n_mu)
        # Initialize mu_filters array
        white_dim = sig.shape[0] * self.ext_fact  # Dimension after extension
        mu_filters = np.zeros((white_dim, n_mu))

        # Extend signals and subtract the mean
        ext_sig = extension(sig, self.ext_fact)

        ext_mean = np.mean(ext_sig, axis=1, keepdims=True) 
        ext_sig -= ext_mean

        # Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method)
        # Loop over each MU
        for i in np.arange(n_mu):
            # Get the optimal MU filter
            w = self.muap_to_filter(muaps[i,:,:], ext_mean, Z)
            # Estimate source
            sources[i,:] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(sources[i,:], fsamp, cluster=self.cluster_method)
            # Store the filter
            mu_filters[:,i] = w

        # Remove bad sources 
        sources, spikes, sil, mu_filters = remove_bad_sources(sources, spikes, sil, mu_filters, 
                                                            threshold=self.sil_th, 
                                                            min_num_spikes=self.min_num_spikes)
        return sources, spikes, sil, mu_filters


    def muap_to_filter(self, muap, ext_mean, Z):
        """
        Get the optimal motor unit filter from the ground truth MUAP.
        Therefore, the MUAP is extended and whitened. The optimal motor unit
        filter corresponds to the column of the extended and whitened MUAP
        that has the highest norm.

        Args:
            MUAP (ndarray): Multichannel MUAP (n_channels x duration)
            Z (ndarray): Whitening matrix

        Returns:
            w (ndarray): Normalized motor unit filter
        """

        # Extend the MUAP
        ext_muap = extension(muap,self.ext_fact) 
        ext_muap -= ext_mean

        # Whiten the MUAP
        white_muap = Z @ ext_muap

        # Find the column with the largest L2 norm and return it as MUAP filter
        col_norms = np.linalg.norm(white_muap, axis=0)
        w = white_muap[:, np.argmax(col_norms)]

        # Normalize w
        w = w/np.linalg.norm(w)

        return(w)
    
    def _write_pipeline_sidecar(self):
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass
    

class basic_cBSS:
    '''
    Class for performing convolutive blind source separation to identify the 
    spiking activity of motor neurons using the fastICA algorithm. 

    
    '''

    def __init__(self, config=None, **kwargs):

        # Default parameters
        self.bandpass = [20, 500]
        self.bandpass_order = 2
        self.notch_frequency = 50
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1
        self.ext_fact = 12
        self.whitening_method = 'ZCA'
        self.whitening_reg  = 'auto'
        self.ica_n_iter = 100
        self.opt_initalization = 'random'
        self.opt_function_exp = 3
        self.opt_max_iter = 100
        self.opt_tol = 1e-4
        self.source_deflation = 'gram-schmidt'
        self.peel_off = 'false'
        self.cluster_method  = 'kmeans'
        self.random_seed = 1909
        self.refinement_loop = True
        self.sil_th = 0.9
        self.cov_th = 0.35
        self.min_num_spikes = 10
        self.match_th = 0.3
        self.match_max_shift = 0.1
        self.match_tol = 0.001
        
        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        # Merge with directly passed keyword arguments (overwrites config)
        params = {**config_dict, **kwargs}

        valid_keys = self.__dict__.keys()

        # Assign all parameters as attributes
        for key, value in params.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

    def set_param(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")   


    def decompose(self, sig, fsamp):
        """
        Run simple decomposition

        Args:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            fsamp (float): Sampling frequency in Hz

        Returns:
            sources (ndarray): Estimated spike responses (n_mu x n_samples)
            spikes (dict): Sample indices of motor neuron discharges 
            sil (ndarray): Pseudo-silhouette scores of the estimated sources
            mu_filters (ndarray): Optimized motor unit filters
        """

        # Initalize random number generator
        rng = np.random.seed(self.random_seed)

        # Bandpass filter signals
        if self.bandpass is not None:
            sig = bandpass_signals(sig, fsamp, 
                                   high_pass = self.bandpass[0], 
                                   low_pass = self.bandpass[1], 
                                   order = self.bandpass_order)

        # Notch filter signals
        if self.notch_frequency is not None:
            sig = notch_signals(sig, fsamp, 
                                nfreq = self.notch_frequency, 
                                dfreq = self.notch_width, 
                                order = self.notch_order, 
                                n_harmonics = self.notch_n_harmonics)

        # Extend signals and subtract the mean and cut the edges
        ext_sig = extension(sig,self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

        # Remove the edges from the exteneded signal
        ext_sig[:,:self.ext_fact*2] = 0
        ext_sig[:,-self.ext_fact*2:] = 0

        # Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method)

        # Initalize the output variables
        sources  = np.zeros((self.ica_n_iter,sig.shape[1]))
        spikes = {i: [] for i in range(self.ica_n_iter)}
        sil = np.zeros(self.ica_n_iter)
        mu_filters = np.zeros((white_sig.shape[0], self.ica_n_iter))

        if self.opt_initalization == 'activity_idx':
            act_idx_histoty = np.array([]) 

        # Loop over each MU
        for i in np.arange(self.ica_n_iter):
            # Initalize 
            if self.opt_initalization == 'random':
                w = np.random.randn(white_sig.shape[0])
            elif self.opt_initalization == 'activity_idx':
                col_norms = np.linalg.norm(white_sig, axis=0)
                col_norms[act_idx_histoty.astype(int)] = 0
                best_idx = np.argmax(col_norms)
                w = white_sig[:, best_idx]
                act_idx_histoty = np.append(act_idx_histoty, best_idx)
            else:
                ValueError('The specified initalization method is not implemented')

            # fastICA fixedpoint optimization
            w, k = self.my_fixed_point_alg(w, white_sig, mu_filters)

            # Predict source and estimate the source quality
            sources[i,:] = w.T @ white_sig
            spikes[i], sil[i] = est_spike_times(sources[i,:], fsamp, cluster=self.cluster_method)
            if len(spikes[i]) > 2:
                isi  = np.diff(spikes[i]/fsamp)
                cov  = np.std(isi) / np.mean(isi)
            else:
                cov = np.inf

            # Refinement loop
            if len(spikes[i]) > 10 and self.refinement_loop:
                w, _, cov = self.mimimize_covisi(w,white_sig, cov, fsamp)
                sources[i,:] = w.T @ white_sig
                spikes[i], sil[i] = est_spike_times(sources[i,:], fsamp, cluster=self.cluster_method)

            # Save the optimized MU filter
            mu_filters[:,i] = w

            # Peel-off the detected source
            if self.peel_off and sil[i] > self.sil_th and cov < self.cov_th:
                white_sig, _ = peel_off(white_sig, spikes[i], win=0.025, fsamp=fsamp) 

        # Remove duplicates        
        sources, spikes, sil, mu_filters = remove_duplicates(sources, spikes, sil, mu_filters, fsamp,
                                                             max_shift=self.match_max_shift,
                                                             tol=self.match_tol,
                                                             threshold=self.match_th)
                                    
        # Remove bad sources 
        sources, spikes, sil, mu_filters  = remove_bad_sources(sources, spikes, sil, mu_filters, 
                                                               threshold=self.sil_th, 
                                                               min_num_spikes=self.min_num_spikes)
       
        return sources, spikes, sil, mu_filters


    def my_fixed_point_alg(self, w, X, B):
        """
        Fixed-point optimization to maximize sparseness of a source signal.

        Args:
            w (np.ndarray): Initial weight vector (n_channels,)
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            B (np.ndarray): Current separation matrix (n_components x n_channels)

        Returns:
            w (np.ndarray): Optimized weight vector
            k (int): Number of iterations taken
        """


        # Define contrast function and its derivative
        # Use g(x)=x*(x**2+epsilon)**((a-1)/2) as smooth approximation of g(x) = sign(x) * abs(x)**a
        epsilon = 1e-3
        a = self.opt_function_exp
        g = lambda x: (epsilon+x**2)**((a-3)/2) * (a*x**2 + epsilon)
        gp = lambda x: (a-1)*x * (epsilon+x**2)**((a-5)/2) * (a*x**2 + 3*epsilon)
        #g = lambda x: x**2
        #gp = lambda x: 2*x

        TOL = self.opt_tol
        delta = np.ones(self.opt_max_iter)
        k = 0

        while delta[k] > TOL and k < self.opt_max_iter - 1:
            w_last = w.copy()

            wTX = w.T @ X  # shape: (n_samples,)
            A = np.mean(gp(wTX))
            w = np.mean(X * g(wTX), axis=1) - A * w  # shape: (n_channels,)

            # Orthogonalization step
            if self.source_deflation == 'projection_deflation':
                w = w - (B @ B.T) @ w
            elif self.source_deflation == 'gram-schmidt':
                w = gram_schmidt(w, B)
            else:
                pass            

            # Normalize
            w = w / np.linalg.norm(w)

            # Convergence criterion
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

        return w, k    
    
    def mimimize_covisi(self, w, X,  cov, fsamp):
        '''
        Iterativly update a motor unit filter given a set of motor neuron 
        spike times as long as the coefficient of variance of the interspike
        intervall decreases.

        Args:
            - w (np.ndarray): Initial weight vector
            - X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            - cov (float): Coefficient of variance of the initial source
            - fsamp (float): Sampling rate in Hz

        Returns: 
            - w (np.ndarray): Optimized weight vector
            - spikes (np.ndarray): Sample indices of motor neuron discharges 
            - cov (float): Coefficient of variance of the optimized source

        '''

        cov_last = cov + 1

        while cov < cov_last:
            source = w.T @ X
            spikes, _ = est_spike_times(source, fsamp)
            cov_last = cov
            isi      = np.diff(spikes/fsamp)
            cov  = np.std(isi) / np.mean(isi)
            w = np.mean(X[:,spikes],axis=1) 
            w = w / np.linalg.norm(w)
         
        return w, spikes, cov
    
    def _write_pipeline_sidecar(self):
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass
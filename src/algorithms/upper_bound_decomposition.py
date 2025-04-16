import numpy as np
from pre_processing import bandpass_signals, notch_signals, extension, whitening
from source_evaluation import est_spike_times

def get_upper_bound(SIG, MUAPs, fsamp, R = 12, wmethod = 'ZCA'):
    """
    Estimate the spike response of motor neurons given the 
    motor unit responses (MUAPs)

    Parameters:
        SIG (ndarray): Input (EMG) signal (n_channels x n_samples)
        MUAPs (ndarray): MUAPs (mu_index x n_channels x duration)
        fsamp (float): Sampling rate of the data (unit: Hz)
        R (int): Extension factor
        wmethod (str): Whitening type 

    Returns:
        ipts (ndarray): Estimated spike responses (n_mu x n_samples)
    """

    n_mu  = MUAPs.shape[0]
    ipts  = np.zeros((n_mu,SIG.shape[1]))
    predicted_spikes = {i: [] for i in range(n_mu)}
    sil = np.zeros(n_mu)

    # Step 1.1: Filter the data -- Take care if that is done also the MUAPs need to filtered accordingly
    if False:
        SIG = bandpass_signals(SIG, fsamp)
        SIG = notch_signals(SIG, fsamp)

    # Step 1.2: Extend signals and subtract the mean
    eSIG = extension(SIG,R)
    eSIG -= np.mean(eSIG, axis=1, keepdims=True)

    # Step 1.3: Whiten the extended signals
    wSIG, Z = whitening(Y=eSIG, method=wmethod)

    # Loop over each MU
    for i in np.arange(n_mu):
        # Get the optimal MU filter
        w = muap_to_filter(MUAPs[i,:,:], Z, R)
        # Estimate source
        ipts[i,:] = w.T @ wSIG
        predicted_spikes[i], sil[i] = est_spike_times(ipts[i,:], fsamp, cluster='kmeans')

    return(ipts, predicted_spikes)


def muap_to_filter(MUAP, Z, R):
    """
    Get the optimal motor unit filter from the ground truth MUAP.
    Therefore, the MUAP is extended and whitened. The optimal motor unit
    filter corresponds to the column of the extended and whitened MUAP
    that has the highest norm.

    Parameters:
        MUAP (ndarray): Multichannel MUAP (n_channels x duration)
        Z (ndarray): Whitening matrix
        R (int): Extension factor

    Returns:
        w (ndarray): Normalized motor unit filter
    """

    # Extend the MUAP
    eMUAP = extension(MUAP,R) 
    # ToDo: Ideally one needs also to take care what was subtracted from the extended signal ...

    # Whiten the MUAP
    wMUAP = Z @ eMUAP

    # Find the column with the largest L2 norm and return it as MUAP filter
    col_norms = np.linalg.norm(wMUAP, axis=0)
    w = wMUAP[:, np.argmax(col_norms)]

    # Normalize w
    w = w/np.linalg.norm(w)

    return(w)
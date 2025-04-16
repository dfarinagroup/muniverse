import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

def est_spike_times(sig, fsamp, cluster = 'kmeans', a = 2):
    """
    Estimate spike indices given a motor unit source signal and compute
    a silhouette-like metric for source quality quantification

    Parameters:
        sig (np.ndarray): Input signal (motor unit source)
        fsamp (float): Sampling rate in Hz
        cluster (string): Clustering method used to identify the spike indices
        a (float): Exponent of assymetric power law 

    Returns:
        est_spikes (np.ndarray): Estimated spike indices
        sil (float): Silhouette-like score (0 = poor, 1 = strong separation)
    """
    sig = np.asarray(sig)

    # Assymetric power law that can be useful for contrast enhancement
    sig = np.sign(sig) * sig**a

    if cluster == 'kmeans':
    
        # Detect peaks with minimum distance of 10 ms
        min_peak_dist = int(round(fsamp * 0.01))
        peaks, _ = find_peaks(sig, distance=min_peak_dist)

        if len(peaks) == 0:
            return np.array([])

        # Get peak values
        peak_vals = sig[peaks].reshape(-1, 1)

        # K-means clustering to separate signal vs. noise
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(peak_vals)
        centroids = kmeans.cluster_centers_.flatten()

        # Spikes are those in the cluster with the higher mean
        spike_cluster = np.argmax(centroids)
        est_spikes = peaks[labels == spike_cluster]

        # Compute within- and between-cluster distances
        D = kmeans.transform(peak_vals)  # Distances to both centroids
        sumd = np.sum(D[labels == spike_cluster, spike_cluster])
        between = np.sum(D[labels == spike_cluster, 1 - spike_cluster])
        
        # Silhouette-inspired score
        denom = max(sumd, between)
        sil = (between - sumd) / denom if denom > 0 else 0.0

    return est_spikes, sil

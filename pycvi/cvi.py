
from typing import List, Sequence, Union, Any, Dict, Tuple
from sklearn.cluster import KMeans
import numpy as np

from .scores import compute_score
from .cluster import compute_cluster_params

def _clusters_data_from_uniform(X, n_clusters, midpoint_w):
    """
    Helper function for "compute_gap_statistics"
    """
    N = len(X)

    # Fit a KMeans model to the sample from a uniform distribution
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    # Sort members into the different clusters and compute their
    # cluster info
    cluster_data = []
    for label_i in range(n_clusters):

        # Members belonging to that clusters
        members = [m for m in range(N) if labels[m] == label_i]
        if members == []:
            raise ValueError('No members in cluster')

        # cluster info
        cluster_info = compute_cluster_params(X[members], midpoint_w)

        cluster_data.append((members, cluster_info))

    return cluster_data

def gap_statistics(
    X : np.ndarray,
    clusters_data: List[Tuple[List[int], Dict]],
    midpoint_w: int = None,
    B: int = 10
) -> float:
    """
    Compute the Gap statistics for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters_data: List of (members, info) tuples
    :type clusters_data: List[Tuple(List[int], Dict)]
    :param midpoint_w: center point of the time window w_t, used as
    reference in case of DTW, defaults to None.
    :type midpoint_w: int, optional
    :param B: Number of uniform samples drawn, defaults to 10
    :type B: int, optional
    :return: The gap statistics
    :rtype: float
    """
    k = len(clusters_data)

    # Compute the log of the within-cluster dispersion for the current clustering
    W_k = compute_score("sum_intra", X, clusters_data)
    wcss = np.log(W_k)

    # Generate B random datasets with the same shape as the input data
    random_datasets = [np.random.rand(X.shape) for _ in range(B)]

    # Compute the log of the within-cluster dispersion for each random dataset
    wcss_rand = []
    for X_rand in random_datasets:
        clusters_data_rand = _clusters_data_from_uniform(X_rand, k, midpoint_w)
        W_k_rand = compute_score("sum_intra", X, clusters_data_rand)
        wcss_rand.append(np.log(W_k_rand))

    # Compute the gap statistic for the current clustering
    gap = np.mean(wcss_rand) - wcss
    return gap

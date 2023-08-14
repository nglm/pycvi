
from typing import List, Sequence, Union, Any, Dict, Tuple
from sklearn.cluster import KMeans
import numpy as np

from .scores import (
    compute_score, f_inertia, f_pdist, f_centroid, f_reduction, f_cdist,
)
from .cluster import compute_cluster_params, compute_center
from .utils import check_dims

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

def _compute_Wk(X, clusters_data):
    """
    Helper function for "compute_gap_statistics"
    """
    # Compute the log of the within-cluster dispersion of the clustering
    nis = [len(cluster) for (cluster, info) in clusters_data]
    d_intra = compute_score("list_intra", X, clusters_data)
    Wk = sum([ni*intra for (ni, intra) in zip(nis, d_intra)])
    return Wk

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

    # Compute the log of the within-cluster dispersion of the clustering
    wcss = np.log(_compute_Wk(X, clusters_data))

    # Generate B random datasets with the same shape as the input data
    random_datasets = [np.random.rand(X.shape) for _ in range(B)]

    # Compute the log of the within-cluster dispersion for each random dataset
    wcss_rand = []
    for X_rand in random_datasets:
        clusters_data_rand = _clusters_data_from_uniform(X_rand, k, midpoint_w)
        wcss_rand.append(np.log(_compute_Wk(X_rand, clusters_data_rand)))

    # Compute the gap statistic for the current clustering
    gap = np.mean(wcss_rand) - wcss
    return gap

def score_function(
    X : np.ndarray,
    clusters_data: List[Tuple[List[int], Dict]],
) -> float:
    """
    Compute the score function for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters_data: List of (members, info) tuples
    :type clusters_data: List[Tuple(List[int], Dict)]
    :return: The score function index
    :rtype: float
    """
    N = len(X)
    k = len(clusters_data)
    global_center = np.expand_dims(compute_center(X), 0)
    dist_kwargs = {"metric" : 'euclidean'}

    bdc = 1/(N*k) * sum([
        # Distance between the centroids and the global centroid
        len(cluster) * f_pdist(
            f_centroid(cluster, info),
            global_center,
            dist_kwargs
        )
        for (cluster, info) in clusters_data
    ])

    wdc = sum([
        f_inertia(cluster, info, dist_kwargs) / len(cluster)
        for (cluster, info) in clusters_data
    ])

    sf = 1 - (1 / (np.exp(np.exp(bdc - wdc))))

    return sf

def hartigan(
    X : np.ndarray,
    clusters_datak1: List[Tuple[List[int], Dict]],
    clusters_datak2: List[Tuple[List[int], Dict]],
) -> float:
    """
    Compute the hartigan index for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters_data: List of (members, info) tuples
    :type clusters_data: List[Tuple(List[int], Dict)]
    :return: The hartigan index
    :rtype: float
    """
    N = len(X)
    k = len(clusters_datak1)
    Wk1 = _compute_Wk(X, clusters_datak1)
    Wk2 = _compute_Wk(X, clusters_datak2)

    hartigan = (Wk1/Wk2 - 1)*(N-k-1)

    return hartigan

def silhouette(
    X : np.ndarray,
    clusters_data: List[Tuple[List[int], Dict]],
) -> float:
    """
    Compute the silhouette score for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters_data: List of (members, info) tuples
    :type clusters_data: List[Tuple(List[int], Dict)]
    :return:The silhouette score
    :rtype: float
    """

    S_i = []

    for i, (c1, info1) in enumerate(clusters_data):

        # Compute a for all x in c1
        a = [f_reduction(f_cdist(c1, x, "average")) for x in c1]

        # Compute b for all x in c1
        b = [np.min([
                f_reduction(f_cdist(c2, x), "average")
                for j, (c2, _) in enumerate(clusters_data) if i !=j
            ]) for x in c1]

        # Silhouette score for cluster c1
        # The usual formula is written a sum divided by the number of x,
        # which is the mean
        S_i.append(np.mean(
            [(b_x - a_x) / (max(b_x, a_x)) for (a_x, b_x) in zip(a, b)]
        ))

    # Silhouette score of the clustering
    # The usual formula is written a sum divided by k, which is the mean
    S = np.mean(S_i)
    return S

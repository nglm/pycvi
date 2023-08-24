
from typing import List, Sequence, Union, Any, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

from .compute_scores import (
    compute_score, f_inertia, f_pdist, get_centroid, reduce, f_cdist,
)
from .cluster import (
    compute_center, generate_uniform, compute_cluster_params, prepare_data,
)
from ._configuration import set_data_shape
from .utils import check_dims

def _clusters_from_uniform(X, n_clusters):
    """
    Helper function for "compute_gap_statistic"
    """
    N = len(X)

    # Fit a KMeans model to the sample from a uniform distribution
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    # Sort members into the different clusters and compute their
    # cluster info
    clusters = []
    for label_i in range(n_clusters):

        # Members belonging to that clusters
        members = [m for m in range(N) if labels[m] == label_i]
        if members == []:
            raise ValueError('No members in cluster')

        clusters.append(members)

    return clusters

def _compute_Wk(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
):
    """
    Helper function for some scores (gap, hartigan, CH, etc.)

    Pooled within-cluster sum of squares around the cluster means (WCSS)
    There are two ways to compute it, using distance to centroid or
    pairwise, we use pairwise, to avoid using barycenters.
    """
    dist_kwargs.setdefault("metric", "sqeuclidean")
    # Compute the log of the within-cluster dispersion of the clustering
    nis = [len(c) for c in clusters]
    d_intra = [f_pdist(X[c], dist_kwargs) for c in clusters]
    Wk = sum([intra/(2*ni) for (ni, intra) in zip(nis, d_intra)])
    return Wk

def _dist_between_centroids(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> List[float]:
    """
    Helper function for some scores (CH, score function, etc.)

    List of distances between cluster centroids and global centroid
    """
    global_center = np.expand_dims(compute_center(X), 0)
    dist = [
        # Distance between the centroids and the global centroid
        f_pdist(
            np.expand_dims(compute_center(X[c]), 0),
            global_center,
            dist_kwargs
        )
        for c in clusters
    ]
    return dist

def gap_statistic(
    X : np.ndarray,
    clusters: List[List[int]],
    B: int = 10
) -> float:
    """
    Compute the Gap statistics for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :param midpoint_w: center point of the time window w_t, used as
    reference in case of DTW, defaults to None.
    :type midpoint_w: int, optional
    :param B: Number of uniform samples drawn, defaults to 10
    :type B: int, optional
    :return: The gap statistics
    :rtype: float
    """
    k = len(clusters)

    # Compute the log of the within-cluster dispersion of the clustering
    wcss = np.log(_compute_Wk(X, clusters))

    # Generate B random datasets with the same shape as the input data
    random_datasets = [np.random.rand(X.shape) for _ in range(B)]

    # Compute the log of the within-cluster dispersion for each random dataset
    wcss_rand = []
    for X_rand in random_datasets:
        clusters_rand = _clusters_from_uniform(X_rand, k)
        wcss_rand.append(np.log(_compute_Wk(X_rand, clusters_rand)))

    # Compute the gap statistic for the current clustering
    gap = np.mean(wcss_rand) - wcss
    return gap

def score_function(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the score function for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The score function index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    dist_kwargs = {"metric" : 'euclidean'}

    nis = [len(c) for c in clusters]

    bdc = 1/(N*k) * sum(
        np.multiply(nis, _dist_between_centroids(X, clusters, dist_kwargs))
    )

    wdc = sum([
        f_inertia(c, dist_kwargs) / len(c)
        for c in clusters
    ])

    sf = 1 - (1 / (np.exp(np.exp(bdc - wdc))))

    return sf

def hartigan(
    X : np.ndarray,
    clustersk1: List[Tuple[List[int], Dict]],
    clustersk2: List[Tuple[List[int], Dict]],
) -> float:
    """
    Compute the hartigan index for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The hartigan index
    :rtype: float
    """
    N = len(X)
    k = len(clustersk1)
    if k == N:
        hartigan = 0.
    else:
        Wk1 = _compute_Wk(X, clustersk1)
        Wk2 = _compute_Wk(X, clustersk2)

        hartigan = (Wk1/Wk2 - 1)*(N-k-1)

    return hartigan

def silhouette(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the silhouette score for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The silhouette score
    :rtype: float
    """

    S_i1 = []
    nis = [len(c) for c in clusters]

    for i1, (c1, _) in enumerate(clusters):
        ni = len(c1)

        # Compute 'a' for all x (=X[m]) in c1 (excluding x in c1)
        a = [(1/nis[i1]) * reduce(f_cdist(X[c1], X[m]), "sum") for m in c1]

        # Compute 'b' for all x (=X[m]) in c1
        b = [np.min([
                reduce(f_cdist(X[c2], X[m]), "mean")
                for i2, (c2, _) in enumerate(clusters) if i1 != i2
            ]) for m in c1]

        # Silhouette score for cluster c1
        # The usual formula is written a sum divided by the number of x,
        # which is the mean
        S_i1.append(np.mean(
            [(b_x - a_x) / (max(b_x, a_x)) for (a_x, b_x) in zip(a, b)]
        ))

    # Silhouette score of the clustering
    # The usual formula is written a sum divided by k, which is the mean
    S = np.mean(S_i1)
    return S

def CH(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
    score_kwargs: dict = {},
) -> float:
    """
    Compute the Calinski–Harabasz (CH) index  for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The CH index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    dist_kwargs.setdefault("metric", "sqeuclidean")

    # If we forget about the (N-k) / (k-1) factor, CH is defined and
    # is equal to 0
    if k == 1:
        CH = 0.
    # Very special case for the case k=0
    elif k == 0:

        # X0 shape: (N, d*w_t) or (N, w_t, d)
        X0 = score_kwargs.get("X0", generate_uniform(X))
        #
        # Option 1: use the centroid of the uniform distribution
        # clusters0 = score_kwargs.get(
        #     "clusters0",
        #     [compute_cluster_params(X0, score_kwargs.get("midpoint_w", 0))]
        # )
        #
        # Option 2: use the original centroid
        #
        # The numerator can be seen as the distance between the global
        # centroid and N singletons uniformly distributed
        # Which can be seen as d(C0, c)
        sep = f_cdist(X0, np.expand_dims(compute_center(X), 0), dist_kwargs)

        # The denominator can be seen as the distance between the
        # original data and its centroid, which would correspond to the
        # denominator or the case k=1 (which is not used because CH(1) =
        # 0)
        # Note that the list has actually only one element
        comp = sum([
            f_inertia(X[c], dist_kwargs) for c in clusters
        ])
        CH = - N * (sep / comp)

    # Normal case
    else:
        nis = [len(c) for c in clusters]
        sep = sum(
            np.multiply(nis, _dist_between_centroids(X, clusters, dist_kwargs))
        )

        comp = sum([
            f_inertia(X[c], dist_kwargs) for c in clusters
        ])

        CH = (N-k) / (k-1) * (sep / comp)
    return CH

def cvi(
    X: np.ndarray,
    cvi: Union[str, callable],
    clustering_model,
    k_range: Sequence = None,
) -> List[float]:
    # Compute all clusterings + compute all scores
    pass

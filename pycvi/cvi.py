
    # :param X: Values of all members.
    # :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    # :param clusters: List of members for each cluster.
    # :type clusters: List[List[int]]
    # :param k: Number of clusters.
    # :type k: int, optional
    # :param dist_kwargs: kwargs for the distance function, defaults to {}
    # :type dist_kwargs: dict, optional
    # :return:
    # :rtype: float

from typing import List, Sequence, Union, Any, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from math import sqrt

from .compute_scores import (
    f_inertia, f_pdist, f_cdist,
)
from .cluster import (
    compute_center, generate_uniform,
)
from .exceptions import NoClusterError, ShapeError

def _clusters_from_uniform(
    X,
    n_clusters,
) -> List[List[int]]:
    """
    Helper function for "compute_gap_statistic"

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param n_clusters: Number of clusters.
    :type n_clusters: int, optional
    :return:
    :rtype: List[List[int]]
    """
    N = len(X)

    # DTW case
    if len(X.shape) == 3:
        model = TimeSeriesKMeans(n_clusters=n_clusters)
    elif len(X.shape) == 2:
        model = KMeans(n_clusters=n_clusters)
    else:
        raise ShapeError("X must have shape (N, T, d) or (N, T*d)")

    # Fit a KMeans model to the sample from a uniform distribution
    labels = model.fit_predict(X)

    # Sort members into the different clusters and compute their
    # cluster info
    clusters = []
    for label_i in range(n_clusters):

        # Members belonging to that clusters
        members = [m for m in range(N) if labels[m] == label_i]
        if members == []:
            raise NoClusterError('No members in cluster')

        clusters.append(members)

    return clusters

def _compute_Wk(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> float:
    """
    Helper function for some scores (gap, hartigan, CH, etc.)

    X is the entire data to be clustered and clusters contains the
    cluster memberships. X can be of shape (N, T, d) or (N, T*d)

    Pooled within-cluster sum of squares around the cluster means (WCSS)
    There are two ways to compute it, using distance to centroid or
    pairwise, we use pairwise, to avoid using barycenters.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: Pooled within-cluster sum of squares around cluster means
        (WCSS)
    :rtype: float
    """
    dist_kwargs.setdefault("metric", "sqeuclidean")
    # Compute the log of the within-cluster dispersion of the clustering
    nis = [len(c) for c in clusters]
    d_intra = [np.sum(f_pdist(X[c], dist_kwargs)) for c in clusters]
    Wk = float(np.sum([intra/(2*ni) for (ni, intra) in zip(nis, d_intra)]))
    return Wk

def _dist_centroids_to_global(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> List[float]:
    """
    Helper function for some scores (CH, score function, etc.)

    List of distances between cluster centroids and global centroid.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: List of distances between cluster centroids and global
        centroid.
    :rtype: List[float]
    """
    global_center = np.expand_dims(compute_center(X), 0)
    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
    dist = [
        # Distance between the centroids and the global centroid
        float(f_cdist(
            center,
            global_center,
            dist_kwargs
        ))
        for center in centers
    ]
    return dist

def _dist_between_centroids(
    X: np.ndarray,
    clusters: List[List[int]],
    all: bool = False,
    dist_kwargs: dict = {},
) -> List[float]:
    """
    Helper function for some scores.

    List of pairwise distances between cluster centroids.

    If there is only one clustering, returns `[0]` (or `[0, 0]` if `all`
    is `True`)

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param all: Should all pairwise distances be returned (all distances
        appear twice) or should distance appear only once?
    :type all: bool, optional
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: List of pairwise distances between cluster centroids.
    :rtype: List[float]
    """
    if len(clusters) == 1:
        dist = [0.]
    else:
        centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
        nested_dist = [
            [
            # pairwise distances between cluster centroids.
                float(f_cdist(
                    center1,
                    center2,
                    dist_kwargs
                ))
                for j, center2 in enumerate(centers[i+1:])
            ] for i, center1 in enumerate(centers[:-1])
        ]

        # Returns the sum of a default value (here []) and an iterable
        # So it flattens the list
        dist = sum(nested_dist, [])
        # If we compute all the pairwise distances, each distance appear twice
    if all:
        dist += dist
    return dist

def gap_statistic(
    X : np.ndarray,
    clusters: List[List[int]],
    k: int = None,
    B: int = 10,
    zero_type: str = "variance",
    return_s: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Compute the Gap statistics for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param B: Number of uniform samples drawn, defaults to 10
    :type B: int, optional
    :param zero_type: zero_type when computing the uniform distribution
    :type zero_type: str, optional
    :param return_s: Should s be returned as well?
    :type return_s: bool, optional
    :return: The gap statistics
    :rtype: Union[float, Tuple[float, float]]
    """
    if k == 0:
        gap = 0.
    else:
        # Compute the log of the within-cluster dispersion of the clustering
        wcss = np.log(_compute_Wk(X, clusters))

        # Generate B random datasets with the same shape as the input data
        # and the same parameters
        random_datasets = generate_uniform(X, zero_type=zero_type, N_zero=B)

        # Compute the log of the within-cluster dispersion for each random dataset
        wcss_rand = []
        for X_rand in random_datasets:
            clusters_rand = _clusters_from_uniform(X_rand, k)
            wcss_rand.append(np.log(_compute_Wk(X_rand, clusters_rand)))

        # Compute the gap statistic for the current clustering
        mean_wcss_rand = np.mean(wcss_rand)
        gap = float(mean_wcss_rand - wcss)

        # To address the case of singletons. Note that gap=0 means that the
        # clustering is irrelevant.
        if gap == -np.inf:
            gap = 0.
            s = 0.
        # To address the case where both are -np.inf which yield gap=nan
        elif mean_wcss_rand == -np.inf and wcss == -np.inf:
            gap = 0.
            s = 0.
        elif return_s:
            # Compute standard deviation of wcss of the references dataset
            sd = np.sqrt( (1/B) * np.sum(
                [(wcss_rand - mean_wcss_rand)**2]
            ))
            # Apply formula
            s = float(np.sqrt(1+(1/B)) * sd)

    if return_s:
        return gap, s
    else:
        return gap

def score_function(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the score function for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :return: The score function index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    dist_kwargs = {"metric" : 'sqeuclidean'}

    nis = [len(c) for c in clusters]

    bdc = 1/(N*k) * np.sum(
        np.multiply(nis, _dist_centroids_to_global(X, clusters, dist_kwargs))
    )

    dist_kwargs = {"metric" : 'euclidean'}
    wdc = np.sum([
        f_inertia(X[c], dist_kwargs) / len(c)
        for c in clusters
    ])

    sf = float(1 - (1 / (np.exp(np.exp(bdc - wdc)))))

    return sf

def hartigan(
    X : np.ndarray,
    clusters: List[List[int]],
    k:int = None,
    clusters_next: List[List[int]] = None,
    X1: np.ndarray = None,
) -> float:
    """
    Compute the hartigan index for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param clusters_next: List of members for each cluster.
    :type clusters_next: List[List[int]]
    :param X1: Values of all members, assuming that k=0 and that X is
        then the values of all members when sampled from a uniform
        distribution.
    :type X1: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :return: The hartigan index
    :rtype: float
    """
    N = len(X)
    # We can't compute using the next clustering, but in any case,
    # we'll have Wk=0
    if k == N:
        hartigan = 0.
    # Because of the factor (N-k-1)
    elif k == N-1:
        hartigan = 0.
    # If we haven't computed the case k+1
    elif clusters_next is None:
        hartigan = 0.
    elif k == 0:
        # X0 shape: (N, d*w_t) or (N, w_t, d)
        if X1 is None:
            l_X0 = generate_uniform(X, zero_type="bounds", N_zero=1)
            X1 = X
        else:
            l_X0 = [X]
        l_Wk = []
        for X0 in l_X0:
            l_Wk.append(_compute_Wk(X0, clusters))
        Wk = np.mean(l_Wk)
        Wk_next = _compute_Wk(X1, clusters)
        # We use the normal formula but with k=1 so that we get the
        hartigan = (Wk/Wk_next - 1)*(N-1-1)
    # Regular case
    else:
        Wk = _compute_Wk(X, clusters)
        Wk_next = _compute_Wk(X, clusters_next)
        hartigan = (Wk/Wk_next - 1)*(N-k-1)

    return hartigan

def silhouette(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the silhouette score for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :return: The silhouette score
    :rtype: float
    """

    S_i1 = []
    nis = [len(c) for c in clusters]

    for i1, c1 in enumerate(clusters):

        # --- Compute 'a' for all x (=X[m]) in c1 (excluding x in c1) --
        # Case in which there is only x in a
        if nis[i1] == 1:
            a = [0]
        # It is usually written as a mean, but since we have to exclude
        # x in c1, we don't get the same number of operations
        else:
            a = [
                (1/(nis[i1]-1)) * np.sum(f_cdist(X[c1], X[m]))
                for m in c1
            ]

        # Compute 'b' for all x (=X[m]) in c1
        b = [np.amin([
                np.mean(f_cdist(X[c2], np.expand_dims(X[m], 0)))
                for i2, c2 in enumerate(clusters) if i1 != i2
            ]) for m in c1]

        # Silhouette score for cluster c1
        # The usual formula is written a sum divided by the number of x,
        # which is the mean
        S_i1.append(np.mean(
            [
                (b_x - a_x) / (np.amax([b_x, a_x]))
                # To address the case of 2 singletons being equal
                if (a_x != 0 or b_x != 0) else 0
                for (a_x, b_x) in zip(a, b)]
        ))

    # Silhouette score of the clustering
    # The usual formula is written a sum divided by k, which is the mean
    S = float(np.mean(S_i1))
    return S

def CH(
    X : np.ndarray,
    clusters: List[List[int]],
    k: int = None,
    X1: np.ndarray = None,
    zero_type: str = "variance",
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the Calinski–Harabasz (CH) index  for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param X1: Values of all members, assuming that k=0 and that X is
        then the values of all members when sampled from a uniform
        distribution.
    :type X1: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The CH index
    :rtype: float
    """
    N = len(X)
    dist_kwargs.setdefault("metric", "sqeuclidean")

    # If we forget about the (N-k) / (k-1) factor, CH is defined and
    # is equal to 0
    if k == 1:
        CH = 0.
    # Very special case for the case k=0
    elif k == 0:

        # X0 shape: (N, d*w_t) or (N, w_t, d)
        if X1 is None:
            X0 = generate_uniform(X, zero_type=zero_type, N_zero=1)[0]
            X1 = X
        else:
            X0 = X
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
        sep = np.sum(
            f_cdist(X0, np.expand_dims(compute_center(X1), 0), dist_kwargs)
        )

        # The denominator can be seen as the distance between the
        # original data and its centroid, which would correspond to the
        # denominator or the case k=1 (which is not used because CH(1) =
        # 0)
        # Note that the list has actually only one element
        comp = np.sum([
            f_inertia(X1[c], dist_kwargs) for c in clusters
        ])
        CH = - N * (sep / comp)
    elif k == N:
        CH=0.
    # Normal case
    else:
        nis = [len(c) for c in clusters]
        sep = np.sum(
            np.multiply(nis, _dist_centroids_to_global(X, clusters, dist_kwargs))
        )

        comp = np.sum([ f_inertia(X[c], dist_kwargs) for c in clusters ])

        CH = (N-k) / (k-1) * (sep / comp)

    CH = float(CH)
    return CH

def MB(
    X : np.ndarray,
    clusters: List[List[int]],
    k: int = None,
    p: int = 2,
    dist_kwargs = {},
) -> float:
    """
    Compute the Maulik-Bandyopadhyay index for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param p: power of the equation
    :type p: int, optional
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The Maulik-Bandyopadhyay index
    :rtype: float
    """
    if k == 1:
        I = 0.
    else:
        N = len(X)
        E1 = sqrt(_compute_Wk(X, [np.arange(N)]))
        Ek = sqrt(_compute_Wk(X, clusters))

        Dk = np.amax(_dist_between_centroids(X, clusters, dist_kwargs))

        I = (1/k * E1/Ek * Dk)**p

    I = float(I)

    return I

def _dis(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Helper function for the SD index, computing the "Dis" term.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :return: The "Dis" term of the SD index.
    :rtype: float
    """
    d_btw_centroids = _dist_between_centroids(
        X, clusters=clusters, all=True, dist_kwargs=dist_kwargs
    )

    dis = float(
        (np.amax(d_btw_centroids) / np.amin(d_btw_centroids))
        * (1 / np.sum(d_btw_centroids))
    )

    return dis

def SD_index(
    X : np.ndarray,
    clusters: List[List[int]],
    alpha: float = None,
    dist_kwargs = {},
) -> float:
    """
    Compute the SD index for a given clustering.

    :param X: Values of all members.
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members for each cluster.
    :type clusters: List[List[int]]
    :param alpha: The constant in the SD index formula (=Dis(k_max)).
    :type alpha: float
    :return: The Maulik-Bandyopadhyay index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    W1 = _compute_Wk(X, [np.arange(N)]) / N

    scat = 1/k * 1/W1 * np.sum(
        [np.mean(f_pdist(X[c], dist_kwargs)**2) for c in clusters]
    )

    if alpha is None:

        d_intra = f_pdist(X, dist_kwargs=dist_kwargs)
        alpha = float(
            (np.amax(d_intra) / np.amin(d_intra))
            * (1 / np.sum(d_intra))
        )
    dis = _dis(X, clusters=clusters, dist_kwargs=dist_kwargs)

    res = float(alpha * scat + dis)
    return res


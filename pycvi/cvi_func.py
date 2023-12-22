"""
Functional API of all implemented CVIs.

These functions are the one on which the CVI classes defined in
:mod:`pycvi.cvi` are based.
"""

from typing import List, Sequence, Union, Any, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from aeon.clustering import TimeSeriesKMeans
import numpy as np
from math import sqrt

from .compute_scores import f_inertia
from .dist import f_pdist, f_cdist
from .cluster import (
    compute_center, generate_uniform, get_clustering
)
from .exceptions import ShapeError

def _clusters_from_uniform(
    X,
    n_clusters,
) -> List[List[int]]:
    """
    Helper function for "compute_gap_statistic"

    :param X: Dataset
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

    # Sort datapoints into the different clusters
    clusters = get_clustering(labels)

    return clusters

def _compute_Wk(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> float:
    """
    Helper function for some CVIs (gap, hartigan, CH, etc.)

    X is the entire data to be clustered and clusters contains the
    cluster memberships. X can be of shape (N, T, d) or (N, T*d)

    Pooled within-cluster sum of squares around the cluster means (WCSS)
    There are two ways to compute it, using distance to centroid or
    pairwise, we use pairwise, to avoid using barycenters.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
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
    Helper function for some CVIs (CH, score function, etc.)

    List of distances between cluster centroids and global centroid.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
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
            dist_kwargs=dist_kwargs
        ))
        for center in centers
    ]
    return dist

def _dist_between_centroids(
    X: np.ndarray,
    clusters: List[List[int]],
    all: bool = False,
    dist_kwargs: dict = {},
) -> Union[List[float], List[float]]:
    """
    Helper function for some CVIs.

    List of pairwise distances between cluster centroids.

    If there is only one clustering, returns `[0]` (or `[0, 0]` if `all`
    is `True`)

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param all: Should all pairwise distances be returned (all distances
        appear twice) or should distance appear only once?
    :type all: bool, optional
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: List of pairwise distances between cluster centroids.
    :rtype: Union[List[float], List[float]]
    """
    if len(clusters) == 1:
        if all:
            dist = [[0.]]
        else:
            dist = [0.]
    else:
        centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
        if all:
            dist = [
                [
                # pairwise distances between cluster centroids.
                    float(f_cdist(
                        center1,
                        center2,
                        dist_kwargs=dist_kwargs
                    )) if i != j else 0.
                    for j, center2 in enumerate(centers)
                ] for i, center1 in enumerate(centers)
            ]
        else:
            nested_dist = [
                [
                # pairwise distances between cluster centroids.
                    float(f_cdist(
                        center1,
                        center2,
                        dist_kwargs=dist_kwargs
                    ))
                    for j, center2 in enumerate(centers[i+1:])
                ] for i, center1 in enumerate(centers[:-1])
            ]

            # Returns the sum of a default value (here []) and an iterable
            # So it flattens the list
            dist = sum(nested_dist, [])
        # If we compute all the pairwise distances, each distance appear twice
    return dist

def _dist_to_centroids(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> List[np.ndarray]:
    """
    Helper function for some CVIs.

    List of distances arrays to cluster centroid.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: List of pairwise distances between cluster centroids.
    :rtype: List[np.ndarray]
    """

    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
    dist = [
        f_cdist(X[cluster], center, dist_kwargs=dist_kwargs)
        for cluster, center in zip(clusters, centers)
    ]
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

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param B: Number of uniform samples drawn, defaults to 10.
    :type B: int, optional
    :param zero_type: Determines how to parametrize the uniform
      distribution to sample from in the case :math:`k=0`, by default
      "variance". Possible options:

      - `"variance"`: the uniform distribution is defined such that it
        has the same variance and mean as the original data.
      - `"bounds"`: the uniform distribution is defined such that it
        has the same bounds as the original data.

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
    k: int =None,
) -> float:
    """
    Compute the score function for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param k: Ignored. Used for compatibility purpose.
    :type k: int
    :return: The score function index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    dist_kwargs = {"metric" : 'sqeuclidean'}

    nis = [len(c) for c in clusters]

    bdc = 1/(N*k) * np.sum(
        np.multiply(nis, _dist_centroids_to_global(
            X, clusters, dist_kwargs=dist_kwargs))
    )

    dist_kwargs = {"metric" : 'euclidean'}
    wdc = np.sum([
        f_inertia(X[c], dist_kwargs=dist_kwargs) / len(c)
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

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param clusters_next: Next clustering (k+1)
    :type clusters_next: List[List[int]]
    :param X1: Dataset. This assumes that k=0 and that X is
        then the values of all datapoints when sampled from a uniform
        distribution.
    :type X1: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :return: The hartigan index
    :rtype: float
    """
    N = len(X)
    # We can't compute using the next clustering, but in any case,
    # we'll have Wk=0
    if k == N:
        return None
    elif k == N-1:
        return None
    # If we haven't computed the case k+1
    elif clusters_next is None:
        return None
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
        res = (Wk/Wk_next - 1)*(N-1-1)
    # Regular case
    else:
        Wk = _compute_Wk(X, clusters)
        Wk_next = _compute_Wk(X, clusters_next)
        if Wk_next == 0:
            res = np.inf
        else:
            res = (Wk/Wk_next - 1)*(N-k-1)

    res = float(res)
    return res

def silhouette(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the silhouette score for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
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

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param k: Number of clusters.
    :type k: int, optional
    :param X1: Dataset. This assumes that k=0 and that X is
      then the values of all datapoints when sampled from a uniform
      distribution.
    :type X1: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param zero_type: Determines how to parametrize the uniform
      distribution to sample from in the case :math:`k=0`, by default
      "variance". Possible options:

      - `"variance"`: the uniform distribution is defined such that it
        has the same variance and mean as the original data.
      - `"bounds"`: the uniform distribution is defined such that it
        has the same bounds as the original data.

    :type zero_type: str, optional
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
        #     [_compute_cluster_params(X0, score_kwargs.get("midpoint_w", 0))]
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

        if comp == 0:
            CH = -np.inf
        else:
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

        if comp == 0:
            CH = np.inf
        else:
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

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
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
    N = len(X)
    if k == 1 or k==N:
        I = 0.
    else:

        dist_kwargs.setdefault("metric", "euclidean")
        E1 = _compute_Wk(X, [np.arange(N)], dist_kwargs=dist_kwargs)
        Ek = _compute_Wk(X, clusters, dist_kwargs=dist_kwargs)

        Dk = np.amax(
            _dist_between_centroids(
                X, clusters, dist_kwargs=dist_kwargs)
            )

        if Ek == 0:
            I = np.inf
        else:
            I = (1/k * E1/Ek * Dk)**p

    I = float(I)

    return I

def _var(
    X : np.ndarray,
    dist_kwargs = {},
) -> np.ndarray:
    """
    Helper function for the SD index, computing the "Var" vector.

    Var is a vector of shape (d,) (or (d*w_t,) if not DTW) of variances

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: "Var" vector of the SD index.
    :rtype: np.ndarray, shape: (d)
    """
    dist_kwargs.setdefault("metric", "sqeuclidean")
    center = np.expand_dims(compute_center(X), 0)
    if len(X.shape) == 2:
        Var = [
            # shape is then (N, 1*w_t) or (N, w_t, 1)
            np.mean(f_cdist(X[:, d:d+1], center[:, d:d+1], dist_kwargs))
            for d in range(X.shape[-1])
        ]
    elif len(X.shape) == 3:
        Var = [
            # shape is then (N, 1*w_t) or (N, w_t, 1)
            np.mean(f_cdist(X[:, :, d:d+1], center[:, :, d:d+1], dist_kwargs))
            for d in range(X.shape[-1])
        ]
    return np.array(Var)

def _dis(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Helper function for the SD index, computing the "Dis" term.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The "Dis" term of the SD index.
    :rtype: float
    """
    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
    d_btw_centroids = _dist_between_centroids(
        X, clusters=clusters, dist_kwargs=dist_kwargs
    )

    # For each center, compute the sum of distances to all other centers
    dis_aux = [
        np.sum(f_cdist(np.concatenate(centers, axis=0), c, dist_kwargs))
        for c in centers
    ]

    dis = float(
        (np.amax(d_btw_centroids) / np.amin(d_btw_centroids))
        * np.sum([1 / d_aux if d_aux != 0 else np.inf for d_aux in dis_aux])
    )

    return dis

def _scat(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Helper function for the SD and SDbw indices.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The "Scat" term of the SD and SDbw indices.
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    # Note that the use of np.linalg.norm is possible here, regardless
    # of whether DTW and/or time series are used because _var always
    # return a vector of shape (d,)
    total_var = np.linalg.norm(_var(X, dist_kwargs=dist_kwargs))

    scat = float(1/k * np.sum([
        np.linalg.norm(_var(X[c], dist_kwargs=dist_kwargs))/total_var
        for c in clusters
    ]))
    return scat

def SD_index(
    X : np.ndarray,
    clusters: List[List[int]],
    alpha: float = None,
    dist_kwargs = {},
) -> float:
    """
    Compute the SD index for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param alpha: The constant in the SD index formula (=Dis(k_max)).
    :type alpha: float
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The SD index
    :rtype: float
    """
    scat = _scat(X, clusters=clusters, dist_kwargs=dist_kwargs)

    # If alpha is None, assume that k_max = N
    if alpha is None:

        alpha_aux = [
            np.sum(f_cdist(X, np.expand_dims(x, 0), dist_kwargs=dist_kwargs))
            for x in X
        ]

        d_intra = f_pdist(X, dist_kwargs=dist_kwargs)
        alpha = float(
            (np.amax(d_intra) / np.amin(d_intra))
            * np.sum(
                [1 / a_aux if a_aux != 0 else np.inf for a_aux in alpha_aux]
            )
        )
    dis = _dis(X, clusters=clusters, dist_kwargs=dist_kwargs)

    res = float(alpha * scat + dis)
    return res

def SDbw_index(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Compute the SDbw index for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The SDbw  index
    :rtype: float
    """
    k = len(clusters)

    scat = _scat(X, clusters=clusters, dist_kwargs=dist_kwargs)

    # Get centroids
    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]

    # Get each (i-j) pair in a flat list and get each pair only once
    nested_ijs = [
        [
            (i, j) for j in range(i+1, k)
        ] for i in range(k-1)
    ]

    ijs = sum(nested_ijs, [])

    # Get the list of midpoints between each pair of cluster center i and j
    # note that u_ij = u_ji and all terms then appear twice
    # We decide to keep each element only once
    u_ijs = [(centers[i] + centers[j])/2 for (i,j) in ijs]

    # Concatenate datapoints corresponding to each u_ij
    # (i.e. datapoints in C_i and C_j)
    X_ijs = [
        np.concatenate((X[clusters[i]], X[clusters[j]]), axis=0)
        for (i,j) in ijs
    ]

    # k (nix1)-arrays of distances to centroids for each cluster
    d_to_centroids = _dist_to_centroids(X, clusters, dist_kwargs=dist_kwargs)

    # Referred as "the average standard deviation of clusters" in the
    # article
    stdev = float(1/k * np.sqrt(np.sum([
        np.linalg.norm(_var(X[c]))
        for c in clusters
    ])))

    # List of (n_ij) arrays of distances to midpoints for each pair of cluster
    # which means the final sum is the double of the list with no
    # duplicates
    d_to_midpoints = [
        f_cdist(X_ij, u_ij, dist_kwargs=dist_kwargs)
        for X_ij, u_ij in zip(X_ijs, u_ijs)
    ]

    # For each pair of cluster ij, count how many datapoints have s
    # distance to u_ij smaller than stdev
    # each d_m is a (n_ij) array of distances to the midpoint u_ij
    densities_ij = [
        np.sum(np.where( d_m <= stdev, np.ones_like(d_m), np.zeros_like(d_m)))
        for d_m in d_to_midpoints
    ]

    # density of each clusters
    # d_i is the (nix1)-array of distances between elements of Ci and ci
    densities_i = [
        np.sum(np.where( d_i <= stdev, np.ones_like(d_i), np.zeros_like(d_i)))
        for d_i in d_to_centroids
    ]
    max_densities_ij = [
        np.amax([densities_i[i], densities_i[j]]) for (i,j) in ijs
    ]

    # The factor 2 is because a term for the pair ij is the same as for ji
    # And we computed only pairs with i<j
    dens_bw = 1/(k*(k-1)) * 2 * np.sum([
        d_ij/max_d_ij if max_d_ij!=0 else np.inf
        for (d_ij, max_d_ij) in zip(densities_ij, max_densities_ij)
    ])

    res = float(scat + dens_bw)
    return res

def dunn(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Compute the Dunn index for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The Dunn index
    :rtype: float
    """
    k=len(clusters)
    N = len(X)

    # This score is not defined for k=N or k=1
    if k == N:
        return 0.

    # Get each (i-j) pair in a flat list and get each pair only once
    nested_ijs = [
        [
            (i, j) for j in range(i+1, k)
        ] for i in range(k-1)
    ]

    ijs = sum(nested_ijs, [])

    # For each ij pair, get minimum distance between x in Ci and y in Cj
    min_dist_between_ij = [
        np.amin(
            f_cdist(X[clusters[i]], X[clusters[j]], dist_kwargs=dist_kwargs)
        ) for (i, j) in ijs
    ]
    # Get the min distance between the 2 closest clusters.
    numerator = np.amin(min_dist_between_ij)

    # For each cluster, get its diameter if it is not a singleton
    diam_i = [
        np.amax(f_pdist(X[c], dist_kwargs=dist_kwargs))
        for c in clusters if len(c) > 1
    ]
    # Get the max diameter of clusters
    denominator = np.amax(diam_i)

    if denominator == 0:
        res = np.inf
    else:
        res = float( numerator / denominator)

    return res

def xie_beni(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Compute the Xie-Beni index for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The Xie-Beni index
    :rtype: float
    """

    N = len(X)
    k = len(clusters)
    if N == k:
        XB = np.inf
    else:

        dist_kwargs.setdefault("metric", "sqeuclidean")
        dist_between_centroids = _dist_between_centroids(
            X, clusters, dist_kwargs=dist_kwargs,
        )

        dist_to_centroids = [
            np.sum(d)
            for d in _dist_to_centroids(X, clusters, dist_kwargs=dist_kwargs)
        ]

        denominator = np.amin(dist_between_centroids)

        if denominator == 0:
            XB = np.inf
        else:
            XB = (1/N) * (np.sum(dist_to_centroids) / denominator)

    XB = float(XB)
    return XB

def xie_beni_star(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """
    Compute the Xie-Beni* (XB*) index for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: The Xie-Beni* (XB*) index
    :rtype: float
    """

    N = len(X)
    k = len(clusters)
    if N == k:
        XB = np.inf
    else:

        dist_kwargs.setdefault("metric", "sqeuclidean")
        dist_between_centroids = _dist_between_centroids(
            X, clusters, dist_kwargs=dist_kwargs,
        )

        dist_to_centroids = [
            np.mean(d)
            for d in _dist_to_centroids(X, clusters, dist_kwargs=dist_kwargs)
        ]

        denominator = np.amin(dist_between_centroids)

        if denominator == 0:
            XB = np.inf
        else:
            XB = np.amax(dist_to_centroids) / denominator

    XB = float(XB)
    return XB

def davies_bouldin(
    X : np.ndarray,
    clusters: List[List[int]],
    p: int = 2,
    dist_kwargs = {},
) -> float:
    """
    Compute the Davies-Bouldin (DB) index for a given clustering.

    :param X: Dataset
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of datapoint indices for each cluster.
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for the distance function, defaults to {}
    :type dist_kwargs: dict, optional
    :return: TheDavies-Bouldin (DB) index
    :rtype: float
    """
    k = len(clusters)

    dist_kwargs_Sis = dist_kwargs.copy()
    dist_kwargs_Sis.setdefault("metric", "minkowski")
    dist_kwargs_Sis.setdefault("p", 1)

    dist_to_centroids = _dist_to_centroids(
        X, clusters, dist_kwargs=dist_kwargs
    )

    nis = [len(c) for c in clusters]
    S_is = [
        (1/ni)**(1/p) * np.sum(d**p)**(1/p)
        for (ni, d) in zip(nis, dist_to_centroids)
    ]

    dist_kwargs_btw_centroids = dist_kwargs.copy()
    dist_kwargs_btw_centroids.setdefault("metric", "minkowski")
    dist_kwargs_btw_centroids.setdefault("p", p)

    dist_between_centroids = _dist_between_centroids(
        X, clusters, all=True, dist_kwargs=dist_kwargs
    )

    DB_aux = [
        np.amax([
            (S_is[i] + S_is[j]) / dist_between_centroids[i][j]
            if dist_between_centroids[i][j] != 0 else np.inf
            for j in range(k)
        ]) for i in range(k)
    ]

    DB = (1/k) * np.sum(DB_aux)
    DB = float(DB)

    return DB
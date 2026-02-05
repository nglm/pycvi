"""Functional API for all implemented CVIs.

These functions are the functional counterparts of the CVI classes in
:mod:`pycvi.cvi`.
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
    """Cluster a uniform sample with k-means.

    This helper is used by the gap statistic to cluster random datasets
    drawn from a uniform distribution.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    list[list[int]]
        Cluster memberships as indices per cluster.

    Raises
    ------
    ShapeError
        If `X` does not have shape (N, T, d) or (N, T*d).
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
    """Compute pooled within-cluster sum of squares (WCSS).

    `X` is the full dataset and `clusters` contains cluster memberships.
    The WCSS is computed from pairwise distances to avoid barycenters.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Pooled within-cluster sum of squares around cluster means.
    """
    # Compute the log of the within-cluster dispersion of the clustering
    nis = [len(c) for c in clusters]

    d_intra = [
        # Sum of squared pairwise distances within cluster c
        np.sum(np.square(f_pdist(X[c], dist_kwargs))) for c in clusters
    ]
    Wk = float(np.sum([intra/(2*ni) for (ni, intra) in zip(nis, d_intra)]))
    return Wk

def _dist_centroids_to_global(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> List[float]:
    """Compute distances between cluster centroids and the global centroid.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    list[float]
        Distances from each cluster centroid to the global centroid.
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
) -> Union[List[float], List[List[float]]]:
    """Compute pairwise distances between cluster centroids.

    If there is only one cluster, returns ``[0]`` (or ``[[0.]]`` when
    `all=True`).

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    all : bool, optional
        Whether to return all pairwise distances (including both
        directions) instead of the upper triangle only.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    list[float] or list[list[float]]
        Pairwise distances between cluster centroids.
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
    squared: bool = False,
    dist_kwargs: dict = {},
) -> List[np.ndarray]:
    """
    Compute (potentially squared) distances from points to their centroid.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    squared : bool, optional
        Whether to return squared distances.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    list[np.ndarray]
        List of distance arrays of shape (N_c, 1) for each cluster.
    """

    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
    dist = [
        f_cdist(X[cluster], center, dist_kwargs=dist_kwargs)
        for cluster, center in zip(clusters, centers)
    ]

    if squared:
        dist = [np.square(d) for d in dist]

    return dist

def _sum_dist_to_centroids(
    X : np.ndarray,
    clusters: List[List[int]],
    squared: bool = False,
    dist_kwargs = {},
) -> List[float]:
    """
    Sum (potentially squared) distances to centroids per cluster.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    squared : bool, optional
        Whether to use squared distances.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    list[float]
        List of summed distances to centroids for each cluster.
    """

    res = [
        np.sum(dist) for dist in _dist_to_centroids(
            X, clusters, squared=squared, dist_kwargs=dist_kwargs
        )
    ]

    return res

def _sum_sum_dist_to_centroids(
    X : np.ndarray,
    clusters: List[List[int]],
    squared: bool = False,
    dist_kwargs = {},
) -> float:
    """
    Sum of summed (potentially squared) distances to centroids.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    squared : bool, optional
        Whether to use squared distances.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Sum of the per-cluster sums of distances to centroids.
    """

    res = float(np.sum( _sum_dist_to_centroids(
        X, clusters, squared=squared, dist_kwargs=dist_kwargs
    )))

    return res

def gap_statistic(
    X : np.ndarray,
    clusters: List[List[int]],
    k: int = None,
    B: int = 10,
    zero_type: str = "variance",
    rng = np.random.default_rng(611),
    return_s: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Compute the gap statistic for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    k : int, optional
        Number of clusters.
    B : int, optional
        Number of uniform samples drawn.
    zero_type : {"variance", "bounds"}, optional
        How to parametrize the uniform distribution when $k=0$.
    rng : numpy.random.Generator, optional
        Random generator used to sample from the uniform distribution.
    return_s : bool, optional
        Whether to return the standard deviation term `s`.

    Returns
    -------
    float or tuple[float, float]
        Gap statistic, and optionally `s` when `return_s=True`.
    """
    if k == 0:
        gap = 0.
    else:
        # Compute the log of the within-cluster dispersion of the clustering
        wcss = np.log(_compute_Wk(X, clusters))

        # Generate B random datasets with the same shape as the input data
        # and the same parameters
        random_datasets = generate_uniform(
            X, zero_type=zero_type, N_zero=B, rng=rng)

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
    dist_kwargs: dict = {},
) -> float:
    """Compute the score function index for a clustering.

    The square-distance version of the score function is used. The
    parameter `k` is accepted for API compatibility but ignored.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    k : int, optional
        Ignored. Present for compatibility.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Score function index.
    """
    N = len(X)
    k = len(clusters)

    nis = [len(c) for c in clusters]

    # List[float]: squared distances between centroids and global centroid
    sq_dist_centroids_to_global = np.square(
        _dist_centroids_to_global(X, clusters, dist_kwargs=dist_kwargs)
    )

    bdc = 1/(N*k) * np.sum( np.multiply(nis, sq_dist_centroids_to_global) )


    # List[float]: sum of squared distances to centroids
    sum_sq_dist_to_centroids = _sum_dist_to_centroids(
        X, clusters, squared=True, dist_kwargs=dist_kwargs
    )

    wdc = np.sum([
        d / ni for (d, ni) in zip(sum_sq_dist_to_centroids, nis)
    ])

    sf = float(1 - (1 / (np.exp(np.exp(bdc - wdc)))))

    return sf

def hartigan(
    X : np.ndarray,
    clusters: List[List[int]],
    k:int = None,
    clusters_next: List[List[int]] = None,
    X1: np.ndarray = None,
    rng = np.random.default_rng(611),
) -> float:
    """Compute the Hartigan index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for the current clustering.
    k : int, optional
        Number of clusters.
    clusters_next : list[list[int]], optional
        Clustering for $k+1$.
    X1 : np.ndarray, optional
        Dataset used when $k=0$ (uniform sample case), representing the
        original data when assuming there is only one cluster.
    rng : numpy.random.Generator, optional
        Random generator used for uniform sampling when needed.

    Returns
    -------
    float or None
        Hartigan index, or ``None`` when undefined for the provided
        inputs.
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
            l_X0 = generate_uniform(X, zero_type="bounds", N_zero=1, rng=rng)
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
    """Compute the silhouette score for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.

    Returns
    -------
    float
        Silhouette score.
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
    rng = np.random.default_rng(611),
    dist_kwargs: dict = {},
) -> float:
    """Compute the Calinski–Harabasz (CH) index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    k : int, optional
        Number of clusters.
    X1 : np.ndarray, optional
        Dataset used when $k=0$ (uniform sample case).
    zero_type : {"variance", "bounds"}, optional
        How to parametrize the uniform distribution when $k=0$.
    rng : numpy.random.Generator, optional
        Random generator used for uniform sampling when needed.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Calinski–Harabasz index.
    """
    N = len(X)

    # If we forget about the (N-k) / (k-1) factor, CH is defined and
    # is equal to 0
    if k == 1:
        CH = 0.
    # Very special case for the case k=0
    elif k == 0:

        # X0 shape: (N, d*w_t) or (N, w_t, d)
        if X1 is None:
            X0 = generate_uniform(X, zero_type=zero_type, N_zero=1, rng=rng)[0]
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
        sep = np.sum(np.square(
            f_cdist(X0, np.expand_dims(compute_center(X1), 0), dist_kwargs)
        ))

        # The denominator can be seen as the distance between the
        # original data and its centroid, which would correspond to the
        # denominator or the case k=1 (which is not used because CH(1) =
        # 0)
        # Note that the list has actually only one element
        comp = _sum_sum_dist_to_centroids(
            X1, clusters, squared=True, dist_kwargs=dist_kwargs
        )

        if comp == 0:
            CH = -np.inf
        else:
            CH = - N * (sep / comp)
    elif k == N:
        CH=0.
    # Normal case
    else:
        nis = [len(c) for c in clusters]

        # List[float]: squared distances between centroids and global centroid
        sq_dist_centroids_to_global = np.square(
            _dist_centroids_to_global(X, clusters, dist_kwargs=dist_kwargs)
        )

        sep = np.sum(
            np.multiply(nis, sq_dist_centroids_to_global)
        )

        comp = _sum_sum_dist_to_centroids(
            X, clusters, squared=True, dist_kwargs=dist_kwargs
        )

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
    """Compute the Maulik–Bandyopadhyay index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    k : int, optional
        Number of clusters.
    p : int, optional
        Exponent used in the index for the distance metric.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Maulik–Bandyopadhyay index.
    """
    N = len(X)
    if k == 1 or k==N:
        I = 0.
    else:

        E1 = _sum_sum_dist_to_centroids(
            X, [np.arange(N)], dist_kwargs=dist_kwargs
        )
        Ek = _sum_sum_dist_to_centroids(
            X, clusters, dist_kwargs=dist_kwargs
        )

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
    """Compute the variance vector used in the SD index.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    np.ndarray
        Variance vector of shape (d,) (or (d*w_t,) for flattened series).
    """
    center = np.expand_dims(compute_center(X), 0)
    if len(X.shape) == 2:
        Var = [
            # shape is then (N, 1*w_t) or (N, w_t, 1)
            np.mean(np.square(
                f_cdist(X[:, d:d+1], center[:, d:d+1], dist_kwargs)
            ))
            for d in range(X.shape[-1])
        ]
    elif len(X.shape) == 3:
        Var = [
            # shape is then (N, 1*w_t) or (N, w_t, 1)
            np.mean(np.square(
                f_cdist(X[:, :, d:d+1], center[:, :, d:d+1], dist_kwargs)
            ))
            for d in range(X.shape[-1])
        ]
    return np.array(Var)

def _dis(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """Compute the dispersion term used in the SD index.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Dispersion ("Dis") term.
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
    """Compute the scatter term used in SD and SDbw indices.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Scatter ("Scat") term.
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
    """Compute the SD index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    alpha : float, optional
        Constant in the SD index formula (defaults to $Dis(k_{max})$).
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        SD index.
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
    """Compute the SDbw index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        SDbw index.
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
    """Compute the Dunn index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Dunn index.
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
    """Compute the Xie–Beni index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Xie–Beni index.
    """

    N = len(X)
    k = len(clusters)
    if N == k:
        XB = np.inf
    else:

        # _dist_between_centroids gives a list of floats
        # sq_dist_between_centroids is then a list floats as well
        sq_dist_between_centroids = [
            d**2 for d in _dist_between_centroids(
                X, clusters, dist_kwargs=dist_kwargs,
            )
        ]

        # float: sum of sum of squared distances to centroids
        sq_dist_to_centroids = _sum_sum_dist_to_centroids(
            X, clusters, squared=True, dist_kwargs=dist_kwargs
        )

        denominator = np.amin(sq_dist_between_centroids)

        # Typically if there are identical centroids denominator is 0
        # then give the worst possible score
        if denominator == 0:
            XB = np.inf
        else:
            XB = (1/N) * (sq_dist_to_centroids / denominator)

    XB = float(XB)
    return XB

def xie_beni_star(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs = {},
) -> float:
    """Compute the Xie–Beni* (XB*) index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Xie–Beni* (XB*) index.
    """

    N = len(X)
    k = len(clusters)
    if N == k:
        XB = np.inf
    else:

        # List[float]: list of distances to centroids
        dist_between_centroids = [
            d ** 2 for d in _dist_between_centroids(
                X, clusters, dist_kwargs=dist_kwargs,
            )
        ]


        numerator = np.amax([
            # _dist_to_centroids gives a list of (N_C, 1) arrays
            np.mean(d) for d in _dist_to_centroids(
                X, clusters, squared=True, dist_kwargs=dist_kwargs
            )
        ])

        denominator = np.amin(dist_between_centroids)

        if denominator == 0:
            XB = np.inf
        else:
            XB = numerator / denominator

    XB = float(XB)
    return XB

def davies_bouldin(
    X : np.ndarray,
    clusters: List[List[int]],
    p: int = 2,
    dist_kwargs = {},
) -> float:
    """Compute the Davies–Bouldin (DB) index for a clustering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, d*w_t) or (N, w_t, d).
    clusters : list[list[int]]
        Indices for each cluster.
    p : int, optional
        Minkowski order when using Euclidean data.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function.

    Returns
    -------
    float
        Davies–Bouldin index.
    """
    k = len(clusters)

    dist_kwargs_Sis = dist_kwargs.copy()

    if len(np.shape(X)) == 2:
        dist_kwargs_Sis.setdefault("metric", "minkowski")
        dist_kwargs_Sis.setdefault("p", p)

    dist_to_centroids = _dist_to_centroids(
        X, clusters, dist_kwargs=dist_kwargs_Sis
    )

    nis = [len(c) for c in clusters]
    S_is = [
        (1/ni)**(1/p) * np.sum(d**p)**(1/p)
        for (ni, d) in zip(nis, dist_to_centroids)
    ]

    dist_kwargs_btw_centroids = dist_kwargs.copy()

    if len(np.shape(X)) == 2:
        dist_kwargs_btw_centroids.setdefault("metric", "minkowski")
        dist_kwargs_btw_centroids.setdefault("p", p)

    dist_between_centroids = _dist_between_centroids(
        X, clusters, all=True, dist_kwargs=dist_kwargs_btw_centroids
    )

    # Compute R_ijs even when i=j
    R_ijs = [
        [
            (S_is[i] + S_is[j]) / dist_between_centroids[i][j]
            if dist_between_centroids[i][j] != 0 else np.inf
            for j in range(k)
        ] for i in range(k)
    ]

    # Remove the case i == j as described in the paper
    R_ijs = [
        [
            R_ijs[i][j] for j in range(k) if i != j
        ] for i in range(k)
    ]

    # Compute R_is = max_j of R_ijs when i != j
    R_is = [
        np.amax(R_ijs) for i in range(k)
    ]

    DB = (1/k) * np.sum(R_is)
    DB = float(DB)

    return DB
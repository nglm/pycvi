"""
Helper module with low and higher level functions to compute CVI values.

.. rubric:: Main function

.. autosummary::
   :template: function.rst

   pycvi.compute_scores.compute_all_scores

"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_soft_dtw_normalized
from typing import List, Sequence, Union, Any, Dict, Tuple

from ._configuration import set_data_shape
from ._utils import _match_dims
from .cluster import (
    compute_center, prepare_data, sliding_window, generate_uniform
)
from .exceptions import InvalidScoreError, InvalidKError, ShapeError

DEFAULT_DIST_KWARGS = {
    "metric" : 'sqeuclidean',
}

def reduce(
    dist: np.ndarray,
    reduction: Union[str, callable] = None,
) -> Union[float, np.ndarray]:
    """
    reduction available: `"sum"`, `"mean"`, `"max"`, `"median"`,
    `"min"`, `""`, `None` or a callable.
    """
    if reduction is not None:
        if reduction == "sum":
            dist = np.sum(dist)
        elif reduction == "average" or reduction == "mean":
            dist = np.mean(dist)
        elif reduction == "median":
            dist = np.median(dist)
        elif reduction == "min":
            dist = np.amin(dist)
        elif reduction == "max":
            dist = np.amax(dist)
        else:
            # Else, assume reduction is a callable
            dist = reduction(dist)
    return dist

def f_pdist(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> np.ndarray:
    """
    Compute the pairwise distance within a group of elements.

    Parameters
    ----------
    cluster : np.ndarray, shape `(N, d)` or `(N, w, d)` if DTW is used.
        A cluster of `N` datapoints.
    dist_kwargs : dict, optional
        kwargs for
        `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_
        , by default {}.

    Returns
    -------
    np.ndarray
        The pairwise distance within the cluster (a condensed matrix).

    Raises
    ------
    ShapeError
        Raised if cluster doesn't have the shape `(N, d)` or `(N, w, d)`
    """
    dims = cluster.shape
    if len(dims) == 2:
        dist = pdist(
            cluster,
            **dist_kwargs
        )
    elif len(dims) == 3:
        # Option 1: Pairwise distances on the entire window using DTW
        (N_c, w_t, d)  = cluster.shape
        dist_square = np.zeros((N_c, N_c))
        for m in range(1, N_c):
            # Fill the upper triangular matrix from the square dist matrix
            dist_square[m-1:m, m:] = cdist_soft_dtw_normalized(
                np.expand_dims(cluster[m-1], 0),
                cluster[m:],
                gamma=1
            )
        # Make this matrix symmetric
        dist_sym = dist_square + dist_square.T - np.diag(np.diag(dist_square))
        # and condense this matrix using squareform
        # squareform gives a square if condensed is given but gives an
        # condensed if a square is given
        dist = squareform(dist_sym)

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
    else:
        msg = (
            f"Can only compute distances between arrays of shapes "
            + f"`(N, d)` or `(N, T, d)`, but got {cluster.shape}"
        )
        raise ShapeError(msg)
    return dist

def f_cdist(
    clusterA: np.ndarray,
    clusterB: np.ndarray,
    dist_kwargs: dict = {},
) -> np.ndarray:
    """
    Compute the distance between two (groups of) elements.

    Parameters
    ----------
    clusterA : np.ndarray
        A cluster of size `NA`.
    clusterB : np.ndarray
        A cluster of size `NB`.
    dist_kwargs : dict, optional
        kwargs for
        `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
        , by default {}.

    Returns
    -------
    np.ndarray, shape `(NA, NB)`
        The pairwise distance matrix between the clusters.

    Raises
    ------
    ShapeError
        Raised if `clusterA` or `clusterB` don't have the shape `(N, d)`
        or `(N, w, d)`.
    """
    clusterA, clusterB = _match_dims(clusterA, clusterB)
    dims = clusterA.shape
    if len(dims) == 2:
        dist = cdist(
            clusterA,
            clusterB,
            **dist_kwargs
        )
    elif len(dims) == 3:
        # Option 1: Pairwise distances on the entire window using DTW
    else:
        msg = (
            f"Can only compute distances between arrays of shapes "
            + f"`(N, d)` or `(N, T, d)`, but got {clusterA.shape} and "
            + f"{clusterB.shape}, reshaped to {dims} to make them match."
        )
        raise ShapeError(msg)

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
        # Note cdist_soft_dtw_normalized should return positive values but
        # somehow doesn't!
    return dist

def f_intra(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the sum of pairwise distance within a group of elements.

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param score_kwargs: kwargs specific for the score.
    :type score_kwargs: dict
    :type cluster_info: dict
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: sum of pairwise distances within the cluster
    :rtype: float
    """
    return float(np.sum(f_pdist(cluster, dist_kwargs)))

def f_inertia(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the inertia within a group of elements.

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param score_kwargs: kwargs specific for the score.
    :type score_kwargs: dict
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: sum of squared distances between each element and the centroid
    :rtype: float
    """
    centroid = np.expand_dims(compute_center(cluster), 0)
    dist = f_cdist(cluster, centroid, dist_kwargs )
    return float(np.sum(dist))

def _f_generalized_var(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the sample generalized variance of ONE cluster

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: generalized variance of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        # a dxd array, representing the sample covariance
        sample_cov = np.cov(cluster, rowvar=False)
        if len(sample_cov.shape) > 1:
            return np.linalg.det(sample_cov)
        else:
            return sample_cov

def _f_med_dev_centroid(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the median deviation around the centroid

    The variance can be seen as the mean deviation around the mean

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the median deviation around the mean of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        centroid = np.expand_dims(compute_center(cluster), 0)
        dist = f_cdist(cluster, centroid, **dist_kwargs )
        return np.median(dist)

def _f_mean_dev_med(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the mean deviation around the median

    The variance can be seen as the mean deviation around the mean

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the mean deviation around the median of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        centroid = np.expand_dims(compute_center(cluster), 0)
        dist = f_cdist(cluster, centroid, **dist_kwargs )
        return np.mean(dist)

def _f_med_dev_med(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the median deviation around the median

    The variance can be seen as the mean deviation around the mean.
    Note that this would not penalize a vertex containing 2 clusters
    with slightly different size

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the mean deviation around the median of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        return np.median(cdist(
            cluster,
            np.median(cluster, axis=0, keepdims=True),
            metric='sqeuclidean'
        ))

def f_diameter(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the diameter of the given cluster

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: the diameter of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        pdist = f_pdist(cluster, dist_kwargs)
        return np.amax(pdist)

def _compute_subscores(
    score_type: str,
    X : np.ndarray,
    clusters: List[List[int]],
    main_score: str,
    f_score,
    dist_kwargs : dict = {},
    score_kwargs : dict = {},
    reduction: str = None,
) -> Union[float, List[float]]:
    """
    Compute the main score of a clustering and its associated subscores

    :param score_type: type of score
    :type score_type: str
    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d)
    :param clusters: List of members, defaults to None
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :param score_kwargs: kwargs specific to the score
    :type score_kwargs: dict
    :param reduction: Type of reduction when computing scores if any
    :type reduction: str
    :return: Score of the given clustering
    :rtype: Union[float, List[float]]
    """
    N = len(X)
    prefixes = ["", "sum_", "mean_", "weighted_"]
    score_tmp = [
            reduce(f_score(X[members], dist_kwargs), reduction)
            for members in clusters
        ]
    if (score_type in [p + main_score for p in prefixes] ):
        # Take the sum
        score = np.sum(score_tmp)
        # Take the mean score by cluster
        if score_type  == "mean_" + main_score:
            score /= len(clusters)
        # Take a weighted mean score by cluster
        elif score_type == "weighted_" + main_score:
            score /= (len(clusters) / N)
    # ------------------------------------------------------------------
    # Return a list of values for each cluster in the clustering
    elif score_type == 'list_' + main_score:
        score = score_tmp
    # ------------------------------------------------------------------
    # Take the median score among all clusters
    elif score_type == 'median_' + main_score:
        score = np.median(score_tmp)
    # ------------------------------------------------------------------
    # Take the max score among all clusters
    elif score_type == 'max_' + main_score:
        score = max(score_tmp)
    # ------------------------------------------------------------------
    # Shouldn't be used: taking min makes no sense
    # Take the min score among all clusters
    elif score_type == 'min_' + main_score:
        score = min(score_tmp)
    else:
        raise InvalidScoreError(
                score_type + " has an invalid prefix."
                + "Please choose a valid score_type"
            )
    return score

def _compute_score(
    score_type: Union[str, callable],
    X: np.ndarray = None,
    clusters: List[List[int]] = None,
    dist_kwargs: dict = {},
    score_kwargs: dict = {},
) -> float :
    """
    Compute the score of a given clustering

    :param score_type: type of score
    :type score_type: Union[str, callable]
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d*w) or (N, w_t, d) optional
    :param clusters: List of members, defaults to None
    :type clusters: List[List[int]]
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :param score_kwargs: kwargs for the CVI.
    :type score_kwargs: dict
    :raises ValueError: If invalid score_type was given
    :return: Score of the given clustering
    :rtype: float
    """

    # TODO: add weights for scores that requires global bounds
    # ------------------------------------------------------------------
    # callable CVI
    if not (type(score_type) == str):
        score = score_type(X, clusters, **score_kwargs)
    else:
        # --------------------------------------------------------------
        # Implemented CVI
        # if score_type == "gap_statistic":
        #     score = gap_statistic(X, clusters_data)
        # elif score_type == "score_function":
        #     score = score_function(X, clusters_data)
        # elif score_type == "silhouette":
        #     score = silhouette(X, clusters_data)
        # elif score_type == "CH":
        #     score = CH(X, clusters_data)
        # --------------------------------------------------------------
        # Inertia-based scores
        if (score_type.endswith("inertia")):
            score = _compute_subscores(
                score_type, X, clusters, "inertia", f_inertia,
                dist_kwargs, score_kwargs
            )
        # within distance-based scores
        elif (score_type.endswith("intra")):
            score = _compute_subscores(
                score_type, X, clusters, "intra", f_intra,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        # Variance based scores
        # Shouldn't be used: use inertia or distortion instead
        # Don't confuse generalized variance and total variation
        # Here it's generalized variance
        elif score_type.endswith("variance"):
            score = _compute_subscores(
                score_type, X, clusters, "variance", _f_generalized_var,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        # Median around mean based scores
        elif score_type.endswith("MedDevCentroid"):
            score = _compute_subscores(
                score_type, X, clusters, "MedDevCentroid", _f_med_dev_centroid,
                dist_kwargs, score_kwargs
            )
        # Median around mean based scores
        elif score_type.endswith("MeanDevMed"):
            score = _compute_subscores(
                score_type, X, clusters, "MeanDevMed", _f_mean_dev_med,
                dist_kwargs, score_kwargs
            )
        # Median around median based scores
        # Shouldn't be used, see _f_med_dev_med
        elif score_type.endswith("MedDevMed"):
            score = _compute_subscores(
                score_type, X, clusters, "MedDevMed", _f_med_dev_med,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        elif score_type.endswith("diameter"):
            score = _compute_subscores(
                score_type, X, clusters, "diameter", f_diameter,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        else:
            raise InvalidScoreError(
                    score_type
                    + " is invalid. Please choose a valid score_type."
                )
    return score

def compute_all_scores(
    score,
    data: np.ndarray,
    clusterings: List[Dict[int, List[List[int]]]],
    transformer = None,
    scaler = StandardScaler(),
    DTW: bool = True,
    time_window: int = None,
    N_zero: int = 10,
    zero_type: str = "bounds",
    score_kwargs: dict = {},
) -> List[Dict[int, float]]:
    """
    Compute all CVI values for the given clusterings.

    If some scores couldn't be computed because of the condition on k
    or because the clustering algorithm used previously didn't converged
    then `scores[t_w][n_clusters] = None`.

    :rtype: List[Dict[int, float]]
    """

    # --------------------------------------------------------------
    # -------- Compute score, cluster params, etc. -----------------
    # --------------------------------------------------------------

    data_copy = set_data_shape(data)
    l_data0 = generate_uniform(data_copy, zero_type=zero_type, N_zero=N_zero)
    (N, T, d) = data_copy.shape
    if scaler is not None:
        scaler.fit(data_copy.reshape(N*T, d))

    if time_window is not None:
        wind = sliding_window(T, time_window)
    else:
        wind = None

    # list of T (if sliding window) or 1 array(s) of shape:
    # (N, T|w_t, d) if DTW
    # (N, (T|w_t)*d) if not DTW
    data_clus = prepare_data(
        data_copy, DTW=DTW, window=wind, transformer=transformer,
        scaler=scaler
    )
    l_data_clus0 = [
        prepare_data(
            data0, DTW=DTW, window=wind, transformer=transformer,
            scaler=scaler
        ) for data0 in l_data0
    ]
    n_windows = len(data_clus)

    # temporary variable to help remember scores
    # scores_t_n[t_w][k] is the score for the clustering assuming k
    # clusters for the extracted time window t_w
    scores_t_n = [{} for _ in range(n_windows)]

    # Note that in this function, "clusterings" corresponds to
    # "clusterings_t_k" in "generate_all_clusterings" and not to
    # "clusters" in cvi functions
    for t_w in range(n_windows):

        for n_clusters in clusterings[t_w].keys():

            # Find cluster membership of each member
            clusters = clusterings[t_w][n_clusters]

            # Take the data used for clustering while taking into
            # account the difference between time step indices
            # with/without sliding window
            X_clus = data_clus[t_w]

            score_kw = score.get_score_kwargs(
                X_clus=X_clus,
                clusterings_t=clusterings[t_w],
                n_clusters=n_clusters,
                score_kwargs=score_kwargs,
            )

            # Special case if the clustering algorithm didn't converge,
            if clusters is None:
                res_score = None
            # Special case k=0: compute average score over N_zero
            # samples
            elif n_clusters == 0:
                l_res_score = []
                for data_clus0 in l_data_clus0:
                    X_clus0 = data_clus0[t_w]
                    try:
                        l_res_score.append(score(
                            X=X_clus0,
                            clusters=clusters,
                            score_kwargs=score_kw,
                        ))
                    except InvalidKError as e:
                        pass
                if l_res_score:
                    res_score = np.mean(l_res_score)
                # If it gave a "InvalidKError" for each sample return None
                else:
                    res_score = None
            else:

                # ------------ Score corresponding to 'n_clusters' ---------
                try:
                    res_score = score(
                        X=X_clus,
                        clusters=clusters,
                        score_kwargs=score_kw,
                    )
                # Ignore if the score was used with a wrong number of clusters
                except InvalidKError as e:
                    res_score = None

            scores_t_n[t_w][n_clusters] = res_score

    return scores_t_n

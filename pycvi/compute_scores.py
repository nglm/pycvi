
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

from ._configuration import set_data_shape
from .utils import match_dims
from .cluster import (
    compute_center, prepare_data, sliding_window, generate_uniform
)

SCORES = [
        'inertia',
        'mean_inertia',  # mean inertia and distortion are the same thing
        'weighted_inertia',
        'max_inertia',
        # ----------
        #'variance',     #TODO: Outdated since DTW
        # ----------
        #"diameter",     #TODO: Outdated since DTW
        #"max_diameter", #TODO: Outdated since DTW
        # ----------
        'MedDevCentroid',
        'mean_MedDevCentroid',
        'max_MedDevCentroid',
]

SUBSCORES = ["", "mean_", "median_", "weighted_", "min_", "max_"]

MAIN_SCORES_TO_MINIMIZE = [
    "inertia",
    #"variance",            #TODO: Outdated since DTW
    "MedDevCentroid",
    #"MeanDevMed",          #TODO: Outdated since DTW
    #"MedDevMed",           #TODO: Outdated since DTW
    #'diameter',            #TODO: Outdated since DTW
]

MAIN_SCORES_TO_MAXIMIZE = []

SCORES_TO_MINIMIZE = [p+s for s in MAIN_SCORES_TO_MINIMIZE for p in SUBSCORES]
SCORES_TO_MAXIMIZE = [p+s for s in MAIN_SCORES_TO_MAXIMIZE for p in SUBSCORES]

SCORES_TO_MINIMIZE += []
SCORES_TO_MAXIMIZE += ["gap_statistic", "CH", "silhouette", "score_function"]

SCORES_MONOTONOUS = [
    "CH", "gap_statistic"
]

SCORES_ABSOLUTE = [
    "silhouette", "score_function"
]

# For monotonous scores only: "does the score increase with croissant number
# of clusters?""
SCORES_INCREASE = [
    "CH", "gap_statistic"
]

# For monotonous scores only: "does the score improve with croissant number
# of clusters?""
SCORES_IMPROVE = [
    "inertia", "mean_inertia", "max_inertia",  "gap_statistic", "CH"
]

DEFAULT_DIST_KWARGS = {
    "metric" : 'sqeuclidean',
}

def reduce(
    dist: np.ndarray,
    reduction: Union[str, callable] = None,
) -> Union[float, np.ndarray]:
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

def get_centroid(
    cluster: np.ndarray,
    cluster_info: dict = None,
) -> np.ndarray:
    """
    The "n_sample" dimension is included in the output

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param cluster_info: info on the center, barycenter, etc. of
        the cluster
    :type cluster_info: dict
    :return: centroid of the cluster
    :rtype: np.ndarray
    """
    dims = cluster.shape
    if len(dims) == 2:
        centroid = cluster_info['center'].reshape(1, -1)
    else:
        centroid = np.expand_dims(cluster_info["barycenter"], 0)
    return centroid

def f_pdist(
    cluster: np.ndarray,
    dist_kwargs: dict = {}
) -> float:
    """
    Compute the pairwise distance within a group of elements

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: pairwise distance within the cluster
    :rtype: float
    """
    dims = cluster.shape
    if len(dims) == 2:
        dist = pdist(
            cluster,
            **dist_kwargs
        )
    if len(dims) == 3:
        # Option 1: Pairwise distances on the entire window using DTW
        (N_c, w_t, d)  = cluster.shape
        dist = np.zeros(N_c)
        for m in range(1, N_c):
            dist[m-1, :N_c-m] = cdist_soft_dtw(
                cluster[m:],
                np.expand_dims(cluster[m-1], 0),
                gamma=1
            )

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
    return dist

def f_cdist(
    clusterA: np.ndarray,
    clusterB: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Compute the distance between two (group of) elements.

    :param clusterA: (NA, d) array, representing a cluster of size NA,
        or (NA, w_t, d) if DTW is used
    :type clusterA: np.ndarray
    :param clusterB: (NB, d) array, representing a cluster of size NB,
        or (NB, w_t, d) if DTW is used
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: pairwise distance within the cluster
    :rtype: float
    """
    clusterA, clusterB = match_dims(clusterA, clusterB)
    dims = clusterA.shape
    if len(dims) == 2:
        dist = cdist(
            clusterA,
            clusterA,
            **dist_kwargs
        )
    if len(dims) == 3:
        # Option 1: Pairwise distances on the entire window using DTW
        dist = cdist_soft_dtw(
            clusterA,
            clusterB,
            gamma=1
        )

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
        # Note cdist_soft_dtw_normalized should return positive values but
        # somehow doesn't!
    return dist

def f_intra(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    score_kwargs: dict = None,
) -> float:
    """
    Compute the sum of pairwise distance within a group of elements

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param score_kwargs: kwargs specific for the score.
    :type score_kwargs: dict
    :type cluster_info: dict
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: pairwise distance within the cluster
    :rtype: float
    """
    return sum(f_pdist(cluster, dist_kwargs))

def f_inertia(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    score_kwargs: dict = None,
) -> float:
    """
    Compute the inertia within a group of elements

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param score_kwargs: kwargs specific for the score.
    :type score_kwargs: dict
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: pairwise distance within the cluster
    :rtype: float
    """
    centroid = np.expand_dims(compute_center(cluster), 0)
    dist = f_cdist(cluster, centroid, **dist_kwargs )
    return sum(dist)

def f_generalized_var(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    score_kwargs: dict = None,
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

def f_med_dev_centroid(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    score_kwargs: dict = None,
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

def f_mean_dev_med(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    score_kwargs: dict = None,
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

def f_med_dev_med(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    score_kwargs: dict = None,
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
    score_kwargs: dict = None,
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
        intra = f_intra(cluster, dist_kwargs)
        return np.amax(intra)

def compute_subscores(
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
            reduce(f_score(X[members], dist_kwargs, score_kwargs), reduction)
            for members in clusters
        ]
    if (score_type in [p + main_score for p in prefixes] ):
        # Take the sum
        score = sum(score_tmp)
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
        raise ValueError(
                score_type + " has an invalid prefix."
                + "Please choose a valid score_type: "
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )
    return score

def compute_score(
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
    :type X: np.ndarray, shape: (N, d*w) optional
    :param clusters_data: List of (members, info) tuples, defaults to None
    :type clusters_data: List[Tuple(List[int], Dict)]
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :param score_kwargs: kwargs for the CVI.
    :type score_kwargs: dict
    :raises ValueError: [description]
    :return: Score of the given clustering
    :rtype: float
    """

    # TODO: add weights for scores that requires global bounds
    # ------------------------------------------------------------------
    # callable CVI
    if not (type(score_type) == str):
        score = score_type(X, clusters, dist_kwargs, score_kwargs)
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
            score = compute_subscores(
                score_type, X, clusters, "inertia", f_inertia,
                dist_kwargs, score_kwargs
            )
        # within distance-based scores
        elif (score_type.endswith("intra")):
            score = compute_subscores(
                score_type, X, clusters, "intra", f_intra,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        # Variance based scores
        # Shouldn't be used: use inertia or distortion instead
        # Don't confuse generalized variance and total variation
        # Here it's generalized variance
        elif score_type.endswith("variance"):
            score = compute_subscores(
                score_type, X, clusters, "variance", f_generalized_var,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        # Median around mean based scores
        elif score_type.endswith("MedDevCentroid"):
            score = compute_subscores(
                score_type, X, clusters, "MedDevCentroid", f_med_dev_centroid,
                dist_kwargs, score_kwargs
            )
        # Median around mean based scores
        elif score_type.endswith("MeanDevMed"):
            score = compute_subscores(
                score_type, X, clusters, "MeanDevMed", f_mean_dev_med,
                dist_kwargs, score_kwargs
            )
        # Median around median based scores
        # Shouldn't be used, see f_med_dev_med
        elif score_type.endswith("MedDevMed"):
            score = compute_subscores(
                score_type, X, clusters, "MedDevMed", f_med_dev_med,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        elif score_type.endswith("diameter"):
            score = compute_subscores(
                score_type, X, clusters, "diameter", f_diameter,
                dist_kwargs, score_kwargs
            )
        # --------------------------------------------------------------
        else:
            raise ValueError(
                    score_type
                    + " is invalid. Please choose a valid score_type: "
                    + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
                )
        return score

def compute_all_scores(
    score,
    data: np.ndarray,
    clusterings: List[Dict[List[List[int]]]],
    transformer = None,
    scaler = StandardScaler(),
    DTW: bool = True,
    time_window: int = None,
    dist_kwargs: dict = {},
    score_kwargs: dict = {},
    verbose: bool = False,
) -> List[Dict[float]]:
    """
    Compute and return all scores

    :rtype: List[Dict[float]]
    """

    # --------------------------------------------------------------
    # -------- Compute score, cluster params, etc. -----------------
    # --------------------------------------------------------------

    data_copy = set_data_shape(data)
    data0 = generate_uniform(data_copy)
    (N, T, d) = data_copy.shape

    if time_window is not None:
        wind = sliding_window(T, time_window)
    else:
        wind = None

    # list of T (if sliding window) or 1 array(s) of shape:
    # (N, T|w_t, d) if DTW
    # (N, (T|w_t)*d) if not DTW
    data_clus = prepare_data(data_copy, DTW, wind, transformer, scaler)
    data_clus0 = prepare_data(data0, DTW, wind, transformer, scaler)
    n_windows = len(data_clus)

    # temporary variable to help remember scores
    # scores_t_n[t_w][k] is the score for the clustering assuming k
    # clusters for the extracted time window t_w
    scores_t_n = [{} for _ in range(n_windows)]

    for t_w in n_windows:

        for n_clusters in clusterings.keys():
            # Take the data used for clustering while taking into account the
            # difference between time step indices with/without sliding window
            if n_clusters == 0:
                X_clus = data_clus0[t_w]
            else:
                X_clus = data_clus[t_w]

            # Find cluster membership of each member
            clusters = clusterings[t_w][n_clusters]

            # ------------ Score corresponding to 'n_clusters' ---------
            score = compute_score(
                score,
                X=X_clus,
                clusters=clusters,
                dist_kwargs=dist_kwargs,
                score_kwargs=score_kwargs
            )

            if verbose:
                print(" ========= {t_w} ========= ")
                msg = "n_clusters: {n_clusters}  ||   score: {score}"
                print(msg)

            scores_t_n[t_w][n_clusters] = score

    return scores_t_n

def better_score(
    score1: float,
    score2: float,
    maximize: bool,
    or_equal: bool = False
) -> bool:
    """
    Determines whether `score1` is indeed better than `score2`.

    If both scores are None, return a ValueError.

    It is assumed that if one (and only one) score is `None` it means
    that it hasn't been reached yet, which means that it is probably
    the best.
    """
    if score1 is None and score2 is None:
        msg = "Better score not determined, both scores are None."
        raise ValueError(msg)
    elif score1 is None:
        return True
    elif score2 is None:
        return False
    elif score1 == score2:
        return or_equal
    elif score1 > score2:
        return maximize
    elif score1 < score2:
        return not maximize
    else:
        msg = "Better score could not be determined: {} | {}".format(
            score1, score2
        )
        raise ValueError(msg)

def argbest(
    scores: List[float],
    maximize: bool,
) -> int:
    """
    Returns index of best score
    """
    if maximize:
        int(np.argmax(scores))
    else:
        int(np.argmin(scores))

def best_score(
    scores: List[float],
    maximize: bool,
) -> float:
    """
    Returns best score
    """
    return scores[argbest(scores, maximize)]

def argworst(
    scores: List[float],
    maximize: bool,
) -> int:
    """
    Returns index of worst score
    """
    if maximize:
        int(np.argmin(scores))
    else:
        int(np.argmax(scores))

def worst_score(
    scores: List[float],
    maximize: bool,
) -> float:
    """
    Returns worst score
    """
    return scores[argworst(scores, maximize)]
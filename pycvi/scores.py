
import numpy as np
from scipy.spatial.distance import cdist, pdist
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

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
        'MedDevMean',
        'mean_MedDevMean',
        'max_MedDevMean',
]

SUBSCORES = ["", "mean_", "median_", "weighted_", "min_", "max_"]

MAIN_SCORES_TO_MINIMIZE = [
    "inertia",
    #"variance",            #TODO: Outdated since DTW
    "MedDevMean",
    #"MeanDevMed",          #TODO: Outdated since DTW
    #"MedDevMed",           #TODO: Outdated since DTW
    #'diameter',            #TODO: Outdated since DTW
]

MAIN_SCORES_TO_MAXIMIZE = []

SCORES_TO_MINIMIZE = [p+s for s in MAIN_SCORES_TO_MINIMIZE for p in SUBSCORES]
SCORES_TO_MAXIMIZE = [p+s for s in MAIN_SCORES_TO_MAXIMIZE for p in SUBSCORES]

DEFAULT_DIST_KWARGS = {
    "metric" : 'sqeuclidean',
}

def f_intra(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
) -> float:
    """
    Compute the pairwise distance within ONE cluster.

    Remember that pdist returns a condensed distance matrix Y. For each
    i and j (where i<j<m), where m is the number of original
    observations. The metric dist(u=X[i], v=X[j]) is computed and stored
    in entry `m * i + j - ((i + 2) * (i + 1)) // 2`.

    :param cluster: (N_c, d) array, representing a cluster of size N_c,
        or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param cluster_info: info on the center, barycenter, etc. of
        the cluster
    :type cluster_info: dict
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: pairwise distance within the cluster
    :rtype: float
    """
    dist_kwargs = {
        **DEFAULT_DIST_KWARGS, **dist_kwargs
    }
    dims = cluster.shape
    if len(dims) == 2:
        score = pdist(
            cluster,
            **dist_kwargs
        )
    if len(dims) == 3:
        barycenter = np.expand_dims(cluster_info["barycenter"], 0)
        # Option 1: Pairwise distances on the entire window using DTW
        Ni = len(cluster)
        score = np.zeros(Ni)
        for m in range(1, Ni):
            score[m-1, :Ni-m] = cdist_soft_dtw(
                cluster[m:],
                cluster[m-1],
                gamma=0.1
            )

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
    return np.sum(score)


def f_inertia(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
) -> float:
    """
    Compute the inertia of ONE cluster

    :param cluster: (N_c, d) array, representing a cluster of size N_c, or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :param cluster_info: info on the center, barycenter, etc. of
        the cluster
    :type cluster_info: dict
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: inertia of that cluster
    :rtype: float
    """
    dist_kwargs = {
        **DEFAULT_DIST_KWARGS, "metric" : 'sqeuclidean',
        **dist_kwargs
    }
    dims = cluster.shape
    if len(dims) == 2:
        score = cdist(
            cluster,
            cluster_info['center'].reshape(1, -1),
            **dist_kwargs
        )
    if len(dims) == 3:
        barycenter = np.expand_dims(cluster_info["barycenter"], 0)
        # Option 1: Pairwise distances on the entire window using DTW
        score = cdist_soft_dtw(
            cluster,
            barycenter,
            gamma=0.1
        )

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
    # Note cdist_soft_dtw_normalized should return positive values but
    # somehow doesn't!
    return np.sum(score)

def f_generalized_var(cluster: np.ndarray, cluster_info: dict = None) -> float:
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

def f_med_dev_mean(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the median deviation around the mean

    The variance can be seen as the mean deviation around the mean

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the median deviation around the mean of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        dims = cluster.shape
        if len(dims) == 2:
            return np.median(cdist(
                cluster,
                cluster_info['center'].reshape(1, -1),
                metric='sqeuclidean'
            ))
        if len(dims) == 3:
            return np.median(cdist_soft_dtw(
                cluster,
                cluster_info["barycenter"],
                gamma=0.1
            ))

def f_mean_dev_med(cluster: np.ndarray, cluster_info: dict = None) -> float:
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
        dims = cluster.shape
        if len(dims) == 2:
            return np.mean(cdist(
                cluster,
                cluster_info['center'].reshape(1, -1),
                metric='sqeuclidean'
            ))
        if len(dims) == 3:
            return np.mean(cdist_soft_dtw(
                cluster,
                cluster_info["barycenter"],
                gamma=0.1
            ))

def f_med_dev_med(cluster: np.ndarray, cluster_info: dict = None) -> float:
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
    cluster_info: dict = None,
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
        intra = f_intra(cluster, cluster_info, dist_kwargs)
        return np.amax(intra)

def compute_subscores(
    score_type: str,
    X : np.ndarray,
    clusters_data: List[Tuple[List[int], Dict]],
    main_score: str,
    f_score,
    dist_kwargs : dict = {},
) -> float:
    """
    Compute the main score of a clustering and its associated subscores

    :param score_type: type of score
    :type score_type: str
    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d)
    :param clusters_data: List of (members, info) tuples, defaults to None
    :type clusters_data: List[Tuple(List[int], Dict)]
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :return: Score of the given clustering
    :rtype: float
    """
    N = len(X)
    prefixes = ["", "sum_", "mean_", "weighted_"]
    score_tmp = [
            f_score(X[members], info, dist_kwargs)
            for members, info in clusters_data
        ]
    if (score_type in [p + main_score for p in prefixes] ):
        # Take the sum
        score = sum(score_tmp)
        # Take the mean score by cluster
        if score_type  == "mean_" + main_score:
            score /= len(clusters_data)
        # Take a weighted mean score by cluster
        elif score_type == "weighted_" + main_score:
            score /= (len(clusters_data) / N)
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
    score_type: str,
    X: np.ndarray = None,
    clusters_data: List[List[int]] = None,
    dist_kwargs: dict = {},
) -> float :
    """
    Compute the score of a given clustering

    :param score_type: type of score
    :type score_type: str
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d*w) optional
    :param clusters_data: List of (members, info) tuples, defaults to None
    :type clusters_data: List[Tuple(List[int], Dict)]
    :param dist_kwargs: kwargs for pdist, cdist, etc.
    :type dist_kwargs: dict
    :raises ValueError: [description]
    :return: Score of the given clustering
    :rtype: float
    """

    # TODO: add weights for scores that requires global bounds

    # ------------------------------------------------------------------
    # Inertia-based scores
    if (score_type.endswith("inertia")):
        score = compute_subscores(
            score_type, X, clusters_data, "inertia", f_inertia,
            dist_kwargs,
        )
    # within distance-based scores
    elif (score_type.endswith("intra")):
        score = compute_subscores(
            score_type, X, clusters_data, "intra", f_intra,
            dist_kwargs
        )
    # ------------------------------------------------------------------
    # Variance based scores
    # Shouldn't be used: use inertia or distortion instead
    # Don't confuse generalized variance and total variation
    # Here it's generalized variance
    elif score_type.endswith("variance"):
        score = compute_subscores(
            score_type, X, clusters_data, "variance", f_generalized_var,
            dist_kwargs
        )
    # ------------------------------------------------------------------
    # Median around mean based scores
    elif score_type.endswith("MedDevMean"):
        score = compute_subscores(
            score_type, X, clusters_data, "MedDevMean", f_med_dev_mean,
            dist_kwargs
        )
    # Median around mean based scores
    elif score_type.endswith("MeanDevMed"):
        score = compute_subscores(
            score_type, X, clusters_data, "MeanDevMed", f_mean_dev_med,
            dist_kwargs
        )
    # Median around median based scores
    # Shouldn't be used, see f_med_dev_med
    elif score_type.endswith("MedDevMed"):
        score = compute_subscores(
            score_type, X, clusters_data, "MedDevMed", f_med_dev_med,
            dist_kwargs
        )
    # ------------------------------------------------------------------
    elif score_type.endswith("diameter"):
        score = compute_subscores(
            score_type, X, clusters_data, "diameter", f_diameter,
            dist_kwargs
        )
    # ------------------------------------------------------------------
    else:
        raise ValueError(
                score_type
                + " is invalid. Please choose a valid score_type: "
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )
    return score

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
    score1: float,
    score2: float,
    maximize: bool,
) -> int:
    """
    Returns index of best score
    """
    if better_score(score1, score2, maximize):
        return 0
    else:
        return 1

def best_score(
    score1: float,
    score2: float,
    maximize: bool,
) -> float:
    """
    Returns best score
    """
    return [score1, score2][argbest(score1, score2, maximize)]

def argworst(
    score1: float,
    score2: float,
    maximize: bool,
) -> int:
    """
    Returns index of worst score
    """
    if better_score(score1, score2, maximize):
        return 1
    else:
        return 0

def worst_score(
    score1: float,
    score2: float,
    maximize: bool,
) -> float:
    """
    Returns worst score
    """
    return [score1, score2][argworst(score1, score2, maximize)]


import numpy as np
from scipy.spatial.distance import cdist, pdist
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

from .utils import match_dims
from .cvi import gap_statistic, silhouette, score_function, CH, hartigan

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
    cluster_info: dict = None,
    dist_kwargs: dict = {}
) -> float:
    """
    Compute the sum of pairwise distance within a group of elements

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
    return sum(f_pdist(cluster, dist_kwargs))

def f_inertia(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
) -> float:
    """
    Compute the inertia within a group of elements

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
    centroid = get_centroid(cluster, cluster_info)
    dist = f_cdist(cluster, centroid, **dist_kwargs )
    return sum(dist)

def f_generalized_var(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
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

def f_med_dev_mean(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
) -> float:
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

def f_mean_dev_med(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
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

def f_med_dev_med(
    cluster: np.ndarray,
    cluster_info: dict = None,
    dist_kwargs: dict = {}
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
    reduction: str = None,
) -> Union[float, List[float]]:
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
    :param reduction: Type of reduction when computing scores if any
    :type reduction: str
    :return: Score of the given clustering
    :rtype: Union[float, List[float]]
    """
    N = len(X)
    prefixes = ["", "sum_", "mean_", "weighted_"]
    score_tmp = [
            reduce(f_score(X[members], info, dist_kwargs), reduction)
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
    score_type: Union[str, callable],
    X: np.ndarray = None,
    clusters_data: List[List[int]] = None,
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
    :param dist_kwargs: kwargs for the CVI.
    :type dist_kwargs: dict
    :raises ValueError: [description]
    :return: Score of the given clustering
    :rtype: float
    """

    # TODO: add weights for scores that requires global bounds
    # ------------------------------------------------------------------
    # callable CVI
    if not (type(score_type) == str):
        score = score_type(
            X, clusters_data, dist_kwargs, score_kwargs
        )
    else:
        # --------------------------------------------------------------
        # Implemented CVI
        if score_type == "gap_statistic":
            score = gap_statistic(X, clusters_data)
        elif score_type == "score_function":
            score = score_function(X, clusters_data)
        elif score_type == "silhouette":
            score = silhouette(X, clusters_data)
        elif score_type == "CH":
            score = CH(X, clusters_data)
        # --------------------------------------------------------------
        # Inertia-based scores
        elif (score_type.endswith("inertia")):
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
        # --------------------------------------------------------------
        # Variance based scores
        # Shouldn't be used: use inertia or distortion instead
        # Don't confuse generalized variance and total variation
        # Here it's generalized variance
        elif score_type.endswith("variance"):
            score = compute_subscores(
                score_type, X, clusters_data, "variance", f_generalized_var,
                dist_kwargs
            )
        # --------------------------------------------------------------
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
        # --------------------------------------------------------------
        elif score_type.endswith("diameter"):
            score = compute_subscores(
                score_type, X, clusters_data, "diameter", f_diameter,
                dist_kwargs
            )
        # --------------------------------------------------------------
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


class Score():

    def __init__(
        self,
        score_function: callable = None,
        maximise: bool = True,
        improve: bool = True,
        score_type: str = "monotonous",
        k_condition: callable = None,
    ) -> None:
        self.function = score_function
        self.maximise = maximise
        self.score_type = score_type
        self.k_condition = k_condition

    def __call__(
        self,
        X: np.ndarray,
        clusters_data: List[Tuple[List[int], Dict]],
        dist_kwargs: dict = {},
        *args: Any, **kwds: Any,
    ) -> Any:
        return self.function(X, clusters_data, dist_kwargs, )

    def is_relevant(self, score, k, score_prev, k_prev) -> bool:
        # A score is always relevant when it is absolute
        if self.score_type == "absolute":
            return True
        else:
            # When relative:
            if self.improve:
                # better "and" greater or worse and lesser
                return (
                    self.better_score(score, score_prev) == k > k_prev
                )
            else:
                # worse "and" greater or better and lesser
                return (
                    self.better_score(score_prev, score) == k_prev > k
                )

    def better_score(
        self,
        score1: float,
        score2: float,
        or_equal: bool = False
    ) -> bool:
        return better_score(score1, score2, self.maximise, or_equal)

    def argbest(
        self,
        score1: float,
        score2: float,
    ) -> int:
        return argbest(score1, score2, self.maximise)

    def best_score(
        self,
        score1: float,
        score2: float,
    ) -> float:

        return best_score(score1, score2, self.maximize)

    def argworst(
        self,
        score1: float,
        score2: float,
    ) -> int:
        return argworst(score1, score2, self.maximize)

    def worst_score(
        self,
        score1: float,
        score2: float,
    ) -> float:
        """
        Returns worst score
        """
        return worst_score(score1, score2, self.maximize)

class Hartigan(Score):

    def __init__(
        self,
        score_type: str = "monotonous"
    ) -> None:

        super().__init__(
            score_function=hartigan,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=0 and k<N)
        )

class CalinskiHarabasz(Score):

    def __init__(
        self,
        score_type: str = "monotonous"
    ) -> None:

        # Note that the case k=1 for the absolute version will always
        # give CH=0
        k_condition = lambda k: (
            k>=0 if score_type == "monotonous" else k>=1
        )

        super().__init__(
            score_function=CH,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition = k_condition,
        )

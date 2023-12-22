"""
Low and high level functions to compute CVI values.

.. rubric:: Main function

.. autosummary::
   :template: function.rst

   pycvi.compute_scores.compute_all_scores

"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import List, Sequence, Union, Any, Dict, Tuple

from.dist import f_cdist, f_pdist, reduce
from ._configuration import set_data_shape
from ._utils import _check_list_of_dict
from .cluster import (
    compute_center, prepare_data, sliding_window, generate_uniform
)
from .exceptions import InvalidScoreError, InvalidKError

DEFAULT_DIST_KWARGS = {
    "metric" : 'sqeuclidean',
}

def f_intra(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Sum of pairwise distances within a group of elements.

    Parameters
    ----------
    cluster : np.ndarray, shape `(N, d)` or `(N, w, d)` if DTW.
        A cluster of size `N`.
    dist_kwargs : dict, optional
        kwargs for
        `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_
        , by default {}.

    Returns
    -------
    float
        The sum of pairwise distances within the cluster.
    """
    return float(np.sum(f_pdist(cluster, dist_kwargs)))

def f_inertia(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Inertia of a group of elements.

    The inertia is defined as the sum of (squared) distances between the datapoints in the the cluster and its centroid.

    Parameters
    ----------
    cluster : np.ndarray, shape `(N, d)` or `(N, w, d)` if DTW.
        A cluster of size `N`.
    dist_kwargs : dict, optional
        kwargs for
        `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
        , by default {}.

    Returns
    -------
    float
        The inertia of the cluster.
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
    Diameter of a group of elements.

    Parameters
    ----------
    cluster : np.ndarray, shape `(N, d)` or `(N, w, d)` if DTW.
        A cluster of size `N`.
    dist_kwargs : dict, optional
        kwargs for
        `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_
        , by default {}.

    Returns
    -------
    float
        The diameter of the cluster.
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
    :param X: Dataset
    :type X: np.ndarray, shape: (N, d)
    :param clusters: List of cluster, defaults to None
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
            reduce(f_score(X[cluster], dist_kwargs), reduction)
            for cluster in clusters
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
    :param X: Dataset, defaults to None
    :type X: np.ndarray, shape: (N, d*w) or (N, w_t, d) optional
    :param clusters: List of cluster, defaults to None
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
    cvi,
    data: np.ndarray,
    clusterings: List[Dict[int, List[List[int]]]],
    transformer: callable = None,
    scaler = StandardScaler(),
    DTW: bool = True,
    time_window: int = None,
    N_zero: int = 10,
    zero_type: str = "bounds",
    cvi_kwargs: dict = {},
    return_list: bool = False,
) -> Union[List[Dict[int, float]], Dict[int, float]]:
    """
    Computes all CVI values for the given clusterings.

    If some scores couldn't be computed because of the condition on
    :math:`k` (:class:`pycvi.exceptions.InvalidKError`) or because the
    clustering algorithm used previously didn't converged
    (:class:`pycvi.exceptions.EmptyClusterError`) then
    ```scores[t_w][n_clusters] = None```.

    Parameters
    ----------
    cvi : an instance of a CVI class.
        The CVI to use to compute all the scores. See
    data : np.ndarray
        Original data. Acceptable input shapes and their corresponding
        output shapes in the PyCVI package:

        - `(N,)` -> `(N, 1, 1)`
        - `(N, d)` -> `(N, 1, d)`
        - `(N, T, d)` -> `(N, T, d)`
    clusterings : List[Dict[int, List[List[int]]]]
        All clusterings for the given range on the number of clusters
        and for the potential sliding windows if applicable.

        ```clusterings_t_k[t_w][k][i]``` is a list of datapoint indices
        contained in cluster :math:`i` for the clustering that assumes
        :math:`k` clusters for the extracted time window :math:`t\_w`.
    transformer : callable, optional
        A potential additional preprocessing step, by default None. If
        None, no transformation is applied on the data
    scaler : A sklearn-like scaler model, optional
        A data scaler, by default
        `StandardScaler() <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_
        . In the case of time
        series data (i.e. :math:`T > 1`), all the time steps of all
        samples of a given feature are aggregated before fitting the
        scaler. If None, no scaling is applied on the data.
    DTW : bool, optional
        Determines if DTW should be used as the distance measure
        (concerns only time series data), by default True.
    time_window : int, optional
        Length of the sliding window (concerns only time-series data),
        by default None. If None, no sliding window is used, and the
        time series is considered as a whole.
    N_zero : int, optional
        Number of uniform distributions sampled, by default 10.
    zero_type : str, optional
        Determines how to parametrize the uniform
        distribution to sample from in the case :math:`k=0`, by default
        "bounds". Possible options:

        - `"variance"`: the uniform distribution is defined such that it
          has the same variance and mean as the original data.
        - `"bounds"`: the uniform distribution is defined such that it
          has the same bounds as the original data.
    cvi_kwargs : dict, optional
        Specific kwargs to give to the CVI, by default {}
    return_list: bool, optional
        Determines whether the output should be forced to be a
        List[Dict], even when no sliding window is used by default False.

    Returns
    -------
    Union[List[Dict[int, float]], Dict[int, float]]
        The computed CVI values for each of the clustering given as
        input.
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
    try:
        clusterings, should_return_list = _check_list_of_dict(clusterings)
    except ValueError as e:
        msg = f"clusterings in compute_all_scores: {e}"
        raise ValueError(msg)
    return_list = return_list or should_return_list

    # Note that in this function, "clusterings" corresponds to
    # "clusterings_t_k" in "generate_all_clusterings" and not to
    # "clusters" in cvi functions
    for t_w in range(n_windows):

        for n_clusters in clusterings[t_w].keys():

            # Find cluster membership of each datapoint
            clusters = clusterings[t_w][n_clusters]

            # Take the data used for clustering while taking into
            # account the difference between time step indices
            # with/without sliding window
            X_clus = data_clus[t_w]

            score_kw = cvi.get_cvi_kwargs(
                X_clus=X_clus,
                clusterings_t=clusterings[t_w],
                n_clusters=n_clusters,
                cvi_kwargs=cvi_kwargs,
            )

            # Special case if the clustering algorithm didn't converge,
            # and raised a EmptyClusterError error.
            if clusters is None:
                res_score = None
            # Special case k=0: compute average score over N_zero
            # samples
            elif n_clusters == 0:
                l_res_score = []
                for data_clus0 in l_data_clus0:
                    X_clus0 = data_clus0[t_w]
                    try:
                        l_res_score.append(cvi(
                            X_clus0,
                            clusters,
                            cvi_kwargs=score_kw,
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
                    res_score = cvi(
                        X_clus,
                        clusters,
                        cvi_kwargs=score_kw,
                    )
                # Ignore if the score was used with a wrong number of clusters
                except InvalidKError as e:
                    res_score = None

            scores_t_n[t_w][n_clusters] = res_score

    # If no sliding window was used, return a Dict, else a List[Dict]
    if return_list:
        return scores_t_n
    else:
        return scores_t_n[0]

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
from .config import set_data_shape
from ._utils import _check_list_of_dict
from .cluster import (
    compute_center, prepare_data, sliding_window,
    generate_uniform
)
from .exceptions import InvalidScoreError, InvalidKError

def f_intra(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Sum of pairwise distances within a group of elements.

    Parameters
    ----------
    cluster : np.ndarray, shape ``(N, d)`` or ``(N, w, d)`` if
    ``ts_dist=True``.
        A cluster of size ``N``.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function. See
        :func:`pycvi.dist.f_pdist` and :func:`pycvi.dist.f_cdist` for
        more information.
    Returns
    -------
    float
        The sum of pairwise distances within the cluster.
    """
    return float(np.sum(f_pdist(cluster, dist_kwargs=dist_kwargs)))

def f_inertia(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
    avg_kwargs: dict = {},
) -> float:
    """
    Inertia of a group of elements.

    The inertia is defined as the sum of (squared) distances between the datapoints in the cluster and its centroid.

    Parameters
    ----------
    cluster : np.ndarray, shape ``(N, d)`` or ``(N, w, d)`` if
    ``ts_dist=True``.
        A cluster of size ``N``.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function. See
        :func:`pycvi.dist.f_pdist` and :func:`pycvi.dist.f_cdist` for
        more information.
    avg_kwargs : dict, optional
        Keyword arguments for the average function. See
        :func:`pycvi.cluster.compute_center` and
        func:`pycvi.cluster.compute_centers` for more information.

    Returns
    -------
    float
        The inertia of the cluster.
    """
    centroid = compute_center(cluster, keepdims=True, avg_kwargs=avg_kwargs)
    dist = f_cdist(cluster, centroid, dist_kwargs=dist_kwargs)
    return float(np.sum(dist))

def f_diameter(
    cluster: np.ndarray,
    dist_kwargs: dict = {},
) -> float:
    """
    Diameter of a group of elements.

    Parameters
    ----------
    cluster : np.ndarray, shape ``(N, d)`` or ``(N, w, d)`` if
    ``ts_dist=True``.
        A cluster of size ``N``.
    dist_kwargs : dict, optional
        Keyword arguments for the distance function. See
        :func:`pycvi.dist.f_pdist` and :func:`pycvi.dist.f_cdist` for
        more information.

    Returns
    -------
    float
        The diameter of the cluster.
    """
    if len(cluster) == 1:
        return 0.
    else:
        pdist = f_pdist(cluster, dist_kwargs=dist_kwargs)
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
    Compute the main score of a clustering and its associated subscores.

    Parameters
    ----------
    score_type : str
        Type of score.
    X : np.ndarray, shape (N, d)
        Dataset.
    clusters : List[List[int]]
        Clustering represented as a list of clusters.
    main_score : str
        Main score suffix used by the score type.
    f_score : callable
        Function used to compute cluster-level scores.
    dist_kwargs : dict, optional
        Keyword arguments for distance computations.
    score_kwargs : dict, optional
        Keyword arguments specific to the score function.
    reduction : str, optional
        Reduction applied to cluster-level values.

    Returns
    -------
    Union[float, List[float]]
        Score of the given clustering.
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
    avg_kwargs: dict = {},
    score_kwargs: dict = {},
) -> float :
    """
    Compute the score of a given clustering.

    Parameters
    ----------
    score_type : Union[str, callable]
        Type of score or callable score function.
    X : np.ndarray, shape (N, d*w) or (N, w_t, d), optional
        Dataset.
    clusters : List[List[int]], optional
        Clustering represented as a list of clusters.
    dist_kwargs : dict, optional
        Keyword arguments for distance computations.
    score_kwargs : dict, optional
        Keyword arguments specific to the CVI.

    Returns
    -------
    float
        Score of the given clustering.

    Raises
    ------
    InvalidScoreError
        If an invalid score type is provided.
    """

    # TODO: add weights for scores that requires global bounds
    # ------------------------------------------------------------------
    # callable CVI
    if not (type(score_type) == str):
        score = score_type(X, clusters, **score_kwargs)
    else:
        # --------------------------------------------------------------
        # Inertia-based scores
        if (score_type.endswith("inertia")):
            # inertia also takes avg_kwargs, so we merge dict
            score_kwargs = avg_kwargs | score_kwargs
            score = _compute_subscores(
                score_type, X, clusters, "inertia", f_inertia,
                dist_kwargs=dist_kwargs, score_kwargs=score_kwargs
            )
        # within distance-based scores
        elif (score_type.endswith("intra")):
            score = _compute_subscores(
                score_type, X, clusters, "intra", f_intra,
                dist_kwargs=dist_kwargs, score_kwargs=score_kwargs
            )
        elif score_type.endswith("diameter"):
            score = _compute_subscores(
                score_type, X, clusters, "diameter", f_diameter,
                dist_kwargs=dist_kwargs, score_kwargs=score_kwargs
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
    ts_dist: bool = True,
    time_window: int = None,
    N_zero: int = 10,
    zero_type: str = "bounds",
    rng = np.random.default_rng(611),
    cvi_kwargs: dict = {},
    return_list: bool = False,
) -> Union[List[List[Dict[int, float]]], List[Dict[int, float]], Dict[int, float]]:
    """
    Computes all CVI values for the given clusterings.

    If some scores couldn't be computed because of the condition on
    :math:`k` (:class:`pycvi.exceptions.InvalidKError`) or because the
    clustering algorithm used previously didn't converged
    (:class:`pycvi.exceptions.EmptyClusterError`) then
    ```scores[t_w][n_clusters] = None```.

    Parameters
    ----------
    cvi : an instance of a CVI class or a CVIAggregator.
        The CVI(s) to use to compute all the scores.
    data : np.ndarray
        Original data. Acceptable input shapes and their corresponding
        output shapes in the PyCVI package:

        - ``(N,)`` -> ``(N, 1, 1)``
        - ``(N, d)`` -> ``(N, 1, d)``
        - ``(N, T, d)`` -> ``(N, T, d)``
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
    ts_dist : bool, optional
        Determines if ts_dist should be used as the distance measure
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
    rng : Union[numpy.random.Generator, int], optional
        The numpy random generator (or seed) to use when sampling from
        random distributions, by default ``np.random.default_rng(611)``
    cvi_kwargs : dict, optional
        Specific kwargs to give to the CVI, by default {}
    return_list: bool, optional
        Determines whether the output should be forced to be a
        List[Dict], even when no sliding window is used by default False.

    Returns
    -------
    Union[List[List[Dict[int, float]]], List[Dict[int, float]],
    Dict[int, float]]
        The computed CVI values for each of the clustering given as
        input.

        The type is:

        - `Dict[int, float]]`: only if a CVI class was used (not a
          CVIAggregator and if no time window was used)
        - `List[List[Dict[int, float]]]`: only if both a CVIAggregator
          was used and a time window
        - `List[Dict[int, float]]`: otherwise, that is to say, if a
          CVIAggregator was used without time window, or if a CVI was
          used with a time window.
    """

    # --------------------------------------------------------------
    # -------- Compute score, cluster params, etc. -----------------
    # --------------------------------------------------------------

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    data_copy = set_data_shape(data)
    l_data0 = generate_uniform(
        data_copy, zero_type=zero_type, N_zero=N_zero, rng=rng
    )
    (N, T, d) = data_copy.shape
    if scaler is not None:
        scaler.fit(data_copy.reshape(N*T, d))

    if time_window is not None:
        wind = sliding_window(T, time_window)
    else:
        wind = None

    # list of T (if sliding window) or 1 array(s) of shape:
    # (N, T|w_t, d) if ts_dist
    # (N, (T|w_t)*d) if not ts_dist
    data_clus = prepare_data(
        data_copy, ts_dist=ts_dist, window=wind, transformer=transformer,
        scaler=scaler
    )
    l_data_clus0 = [
        prepare_data(
            data0, ts_dist=ts_dist, window=wind, transformer=transformer,
            scaler=scaler
        ) for data0 in l_data0
    ]
    n_windows = len(data_clus)

    try:
        clusterings, was_list = _check_list_of_dict(clusterings)
    except ValueError as e:
        msg = f"clusterings in compute_all_scores: {e}"
        raise ValueError(msg)
    return_list = return_list or was_list

    # Special case with aggregators
    if hasattr(cvi, '_is_aggregator') and cvi._is_aggregator:
        list_cvi = cvi.cvis
        list_cvi_kwargs = cvi.cvi_kwargs
        is_aggregator = True
    else:
        list_cvi = [cvi]
        list_cvi_kwargs = [cvi_kwargs]
        is_aggregator = False

    # temporary variable to help remember scores
    # scores_i_t_n[[i]t_w][k] is the score for the clustering assuming k
    # clusters for the extracted time window t_w with CVI i
    scores_i_t_n = [
        [{} for _ in range(n_windows)] for _ in range(len(list_cvi))
    ]

    # Note that in this function, "clusterings" corresponds to
    # "clusterings_t_k" in "generate_all_clusterings" and not to
    # "clusters" in cvi functions
    for i in range(len(list_cvi)):
        for t_w in range(n_windows):
            for n_clusters in clusterings[t_w].keys():

                # Find cluster membership of each datapoint
                clusters = clusterings[t_w][n_clusters]

                # Take the data used for clustering while taking into
                # account the difference between time step indices
                # with/without sliding window
                X_clus = data_clus[t_w]

                score_kw = list_cvi[i].get_cvi_kwargs(
                    X_clus=X_clus,
                    clusterings_t=clusterings[t_w],
                    n_clusters=n_clusters,
                    cvi_kwargs=list_cvi_kwargs[i],
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
                            l_res_score.append(list_cvi[i](
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
                        res_score = list_cvi[i](
                            X_clus,
                            clusters,
                            cvi_kwargs=score_kw,
                        )
                    # Ignore if the score was used with a wrong number
                    # of clusters
                    except InvalidKError as e:
                        res_score = None

                scores_i_t_n[i][t_w][n_clusters] = res_score

    # -------------------------- Fix output type ------------------------------

    # If no sliding window was used, each element is Dict otherwise List[Dict]
    if return_list:
        output = scores_i_t_n
    else:
        output = [score[0] for score in scores_i_t_n]
    # If CVIAggregator was used return a list of Dict or a list of List[Dict]
    # else return a Dict or a List[Dict]
    if not is_aggregator:
        output = output[0]

    return output

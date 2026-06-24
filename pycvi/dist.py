"""
Low-level distance functions for (non-) time-series data.

"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from aeon.distances import pairwise_distance
from typing import List, Sequence, Union, Any, Dict, Tuple
from ._utils import _match_dims
from .exceptions import ShapeError
from .config import default_ts_distance_kwargs

def reduce(
    dist: np.ndarray,
    reduction: Union[str, callable] = None,
) -> Union[float, np.ndarray]:
    """
    Applies a given operation on a distance matrix.

    reduction available: `"sum"`, `"mean"`, `"max"`, `"median"`,
    `"min"`, `""`, `None` or a callable.

    Parameters
    ----------
    dist : np.ndarray,
        A distance matrix, either condensed (if pdist) or not (if
        cdist).
    reduction : Union[str, callable], optional
        The type of reduction to apply to the distance matrix, by
        default None.

    Returns
    -------
    Union[float, np.ndarray]
        The result of applying the reduction on the distance matrix.
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
    Pairwise distances within a group of elements.

    The user can provide a custom callable together with its kwargs in
    the ``dist_kwargs`` parameter. To provide a callable, use the key
    ``"CALLABLE"``, otherwise the default distance function will be used,
    which depends on the type of data (time series or static).

    In the case of static data
    ---------------------------

    Calls `scipy.spatial.distance.pdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_,
    which offers a wide range of distances and parameters, all of them
    described in `scipy.spatial.distance.pdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_.

    By default, PyCVI relies on scipy's default parameters, which means
    that the actual distance used is the `euclidean distance
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean>`_.


    In the case of time series data
    --------------------------------

    Calls `aeon.distances.pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.pairwise_distance.html>`_
    which offers a wide range of distances and parameters. See
    `aeon.distances
    <https://www.aeon-toolkit.org/en/latest/api_reference/distances.html>`_
    for an overview of the distance functions available in `aeon` as
    well as their parameters. For each available distance function, you
    can also use a short name as described in
    `aeon.distances.get_pairwise_distance_function
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.get_pairwise_distance_function.html#aeon.distances.get_pairwise_distance_function>`_.

    By default, PyCVI uses the following ``dist_kwargs`` value:
    ``{"method" : "msm"}``, which means that the actual distance used is
    MSM, implemented in `aeon` in the
    `aeon.distances.msm_pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.msm_pairwise_distance.html#aeon.distances.msm_pairwise_distance>`_
    function. See :func:`pycvi.config.default_ts_distance_kwargs` for
    more information about default distance kwargs used in PyCVI.

    Parameters
    ----------
    cluster : np.ndarray, shape ``(N, d)`` or ``(N, w, d)`` if ``ts_dist=True``.
        A cluster of ``N`` datapoints.
    dist_kwargs : dict, optional
        Additional kwargs for the distance function.

    Returns
    -------
    np.ndarray
        The pairwise distance within the cluster (a condensed matrix).

    Raises
    ------
    ShapeError
        Raised if cluster doesn't have the shape ``(N, d)`` or
        ``(N, w, d)``. See :func:`pycvi.config.set_data_shape` for more
        information on acceptable shapes.
    """
    dims = cluster.shape
    # ------------------- Static data ------------------------------
    if len(dims) == 2:

        # Case where the user provides a custom callable
        if "CALLABLE" in dist_kwargs:
            distance_function = dist_kwargs.pop("CALLABLE")
        # Uses scipy.spatial.distance.pdist by default
        else:
            distance_function = pdist

        dist = distance_function(
            cluster,
            **dist_kwargs
        )
    # ------------------ Time series data ------------------------------
    elif len(dims) == 3:
        # Option 1: Pairwise distances on the entire window using aeon
        (N_c, w_t, d) = cluster.shape

        dist_kwargs_final = default_ts_distance_kwargs(dist_kwargs)

        if "CALLABLE" in dist_kwargs_final:
            distance_function = dist_kwargs_final.pop("CALLABLE")
        else:
            distance_function = pairwise_distance

        dist_square = distance_function(
            np.swapaxes(cluster, 1, 2),
            None,
            **dist_kwargs_final,
        )
        # and condense this matrix using squareform
        # squareform gives a square if condensed is given but gives an
        # condensed if a square is given
        dist = squareform(dist_square)

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each datapoint in the cluster
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
    Distances between two (groups of) elements.

    The user can provide a custom callable together with its kwargs in
    the ``dist_kwargs`` parameter. To provide a callable, use the key
    ``"CALLABLE"``, otherwise the default distance function will be used,
    which depends on the type of data (time series or static).

    In the case of static data
    ---------------------------

    Calls `scipy.spatial.distance.cdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_,
    which offers a wide range of distances and parameters, all of them
    described in `scipy.spatial.distance.cdist
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_.

    By default, PyCVI relies on scipy's default parameters, which means
    that the actual distance used is the `euclidean distance
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean>`_.


    In the case of time series data
    --------------------------------

    Calls `aeon.distances.pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.pairwise_distance.html>`_
    which offers a wide range of distances and parameters. See
    `aeon.distances
    <https://www.aeon-toolkit.org/en/latest/api_reference/distances.html>`_
    for an overview of the distance functions available in `aeon` as
    well as their parameters. For each available distance function, you
    can also use a short name as described in
    `aeon.distances.get_pairwise_distance_function
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.get_pairwise_distance_function.html#aeon.distances.get_pairwise_distance_function>`_.

    By default, PyCVI uses the following ``dist_kwargs`` value:
    ``{"method" : "msm"}``, which means that the actual distance used is
    MSM, implemented in `aeon` in the
    `aeon.distances.msm_pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.msm_pairwise_distance.html#aeon.distances.msm_pairwise_distance>`_
    function. See :func:`pycvi.config.default_ts_distance_kwargs` for
    more information about default distance kwargs used in PyCVI.

    Parameters
    ----------
    clusterA : np.ndarray
        A cluster of size `NA`.
    clusterB : np.ndarray
        A cluster of size `NB`.
    dist_kwargs : dict, optional
        Additional kwargs for the distance function.

    Returns
    -------
    np.ndarray, shape `(NA, NB)`
        The pairwise distance matrix between the clusters.

    Raises
    ------
    ShapeError
        Raised if ``clusterA`` or ``clusterB`` don't have the shape
        ``(N, d)`` or ``(N, w, d)``.
    """
    clusterA, clusterB = _match_dims(clusterA, clusterB)
    dims = clusterA.shape
    # ------------------- Static data ------------------------------
    if len(dims) == 2:

        # Case where the user provides a custom callable
        if "CALLABLE" in dist_kwargs:
            distance_function = dist_kwargs.pop("CALLABLE")
        # Uses scipy.spatial.distance.cdist by default
        else:
            distance_function = cdist

        dist = distance_function(
            clusterA,
            clusterB,
            **dist_kwargs
        )
    # ------------------ Time series data ------------------------------
    elif len(dims) == 3:

        dist_kwargs_final = default_ts_distance_kwargs(dist_kwargs)

        if "CALLABLE" in dist_kwargs_final:
            distance_function = dist_kwargs_final.pop("CALLABLE")
        else:
            distance_function = pairwise_distance

        # Option 1: Pairwise distances on the entire window using aeon
        dist = distance_function(
            np.swapaxes(clusterA, 1, 2),
            np.swapaxes(clusterB, 1, 2),
            **dist_kwargs_final,
        )
    else:
        msg = (
            f"Can only compute distances between arrays of shapes "
            + f"`(N, d)` or `(N, T, d)`, but got {clusterA.shape} and "
            + f"{clusterB.shape}, reshaped to {dims} to make them match."
        )
        raise ShapeError(msg)

    return dist

def time_series_metric_with_sklearn(X, dist_kwargs={}, d=1, w_t=None):
    """
    Allow to use time-series metrics with (some) sklearn models.

    Some `sklearn` models have a ``"metric"`` parameter that accepts a
    callable, see for example `sklearn.cluster.AgglomerativeClustering
    <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_.
    We can then use a metric specifically designed for time-series such
    as those defined in `aeon`, provided that we call the distance
    function on a reshaped version of the data. Indeed, `sklearn` only
    allows data of shape ``(N, d)`` (or ``(N, d*T)``) while time-series
    distances in `aeon` require data of shape ``(N, T, d)``.

    Thus, this present function reshapes the data accordingling on the
    fly such that one can use time series distances with (some) sklearn
    models.

    To be able to do the reshaping, it is important to correctly provide
    the original ``N`` and ``d`` values, as if the following happened:

    1. The data ``X`` was originally of shape ``(N, T, d)`` (Starting
       point)
    2. ``X`` was reshaped to ``(N, T*d)`` to match ``sklearn``
       requirements (typically using ``X = np.reshape(X, (N, -1))``) (To
       be done by the user before using the sklearn (or sklearn-like)
       model)
    3. Inside the call of the sklearn-like model, ``X`` is reshaped back
       to ``(N, T, d)`` to match ``aeon`` requirements (part that is
       done by this function)

    See :func:`pycvi.config.default_ts_distance_kwargs` for more
    information about default distance kwargs used in PyCVI and see See
    :func:`pycvi.dist.f_pdist` for more information about distances with
    time series data in PyCVI.

    For a full example of this function, see TODO

    """
    dims = X.shape
    N = len(X)
    if T is None:
        T = dims[-1]
    # Go from (N, T*d) to (N, T, d)
    # assuming we had either (N, T*1) or (N, T, d) to begin with
    shape = (X, T, d)


    def _aux(X, dist_kwargs={}):
        X_dis = np.reshape(X, shape)
        return f_pdist(X_dis, dist_kwargs=dist_kwargs)

    return _aux
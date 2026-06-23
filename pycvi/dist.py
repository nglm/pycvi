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
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.pairwise_distance.html#dtw-pairwise-distance>`_
    which offers a wide range of distances and parameters. See
    `aeon.distances
    <https://www.aeon-toolkit.org/en/latest/api_reference/distances.html>`_
    for an overview of the distance functions available in `aeon` as
    well as their parameters. For each available distance function, you
    can also use a short name as described in
    `aeon.distances.get_pairwise_distance_function
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.get_pairwise_distance_function.html#aeon.distances.get_pairwise_distance_function>`_.

    By default, PyCVI uses the following ``dist_kwargs`` value:
    ``{"method" : "dtw", window : 0.2}``, which means that the actual
    distance used is DTW, implemented in `aeon` in the
    `aeon.distances.dtw_pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.dtw_pairwise_distance.html#aeon.distances.dtw_pairwise_distance>`_
    function. See :func:`pycvi.config.default_ts_distance_kwargs` for
    more information about default distance kwargs used in PyCVI.

    Parameters
    ----------
    cluster : np.ndarray, shape ``(N, d)`` or ``(N, w, d)`` if DTW is
    used.
        A cluster of `N` datapoints.
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
    if len(dims) == 2:
        dist = pdist(
            cluster,
            **dist_kwargs
        )
    elif len(dims) == 3:
        # Option 1: Pairwise distances on the entire window using DTW
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
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.pairwise_distance.html#dtw-pairwise-distance>`_
    which offers a wide range of distances and parameters. See
    `aeon.distances
    <https://www.aeon-toolkit.org/en/latest/api_reference/distances.html>`_
    for an overview of the distance functions available in `aeon` as
    well as their parameters. For each available distance function, you
    can also use a short name as described in
    `aeon.distances.get_pairwise_distance_function
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.get_pairwise_distance_function.html#aeon.distances.get_pairwise_distance_function>`_.

    By default, PyCVI uses the following ``dist_kwargs`` value:
    ``{"method" : "dtw", window : 0.2}``, which means that the actual
    distance used
    is DTW, implemented in `aeon` in the
    `aeon.distances.dtw_pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.dtw_pairwise_distance.html#aeon.distances.dtw_pairwise_distance>`_
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
    if len(dims) == 2:
        dist = cdist(
            clusterA,
            clusterB,
            **dist_kwargs
        )
    elif len(dims) == 3:

        dist_kwargs_final = default_ts_distance_kwargs(dist_kwargs)

        if "CALLABLE" in dist_kwargs_final:
            distance_function = dist_kwargs_final.pop("CALLABLE")
        else:
            distance_function = pairwise_distance

        # Option 1: Pairwise distances on the entire window using DTW
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

        # Option 2: Pairwise distances between the midpoint of the barycenter
        # and the corresponding time step for each member in the cluster
        # TODO
        # Note cdist_soft_dtw_normalized should return positive values but
        # somehow doesn't!
    return dist
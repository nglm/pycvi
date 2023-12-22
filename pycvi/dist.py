"""
Low-level distance functions for (non-) time-series data.

"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from aeon.distances import dtw_distance, dtw_pairwise_distance
from typing import List, Sequence, Union, Any, Dict, Tuple
from ._utils import _match_dims
from .exceptions import ShapeError

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
        (N_c, w_t, d) = cluster.shape

        dist_square = dtw_pairwise_distance(
            np.swapaxes(cluster, 1, 2),
            None,
            # itakura_max_slope=0.1,
            window=0.2,
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
        dist = dtw_pairwise_distance(
            np.swapaxes(clusterA, 1, 2),
            np.swapaxes(clusterB, 1, 2),
            # itakura_max_slope=0.1,
            window=0.2,
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
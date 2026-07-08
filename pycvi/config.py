"""Configure default parameters and shapes"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from numpy.random import RandomState, Generator
from typing import List, Sequence, Union, Any, Dict, Tuple
from .exceptions import ShapeError

def set_data_shape(X: np.ndarray) -> np.ndarray:
    """
    Returns a copy of the data but with the right shape (N, T, d)

    Acceptable input shapes and their corresponding output shapes:

    - `(N,)` -> `(N, 1, 1)`
    - `(N, d)` -> `(N, 1, d)`
    - `(N, T, d)` -> `(N, T, d)`

    Parameters
    ----------
    X : np.ndarray
        Original data

    Returns
    -------
    np.ndarray
        The same data but re-shaped to match the requirements of the
        PyCVI package

    Raises
    ------
    ShapeError
        Raised if only one sample was given.
    ShapeError
        Raised if an invalid shape dimensions was provided
        (1<=dimensions<=3)
    """
    X_copy = np.copy(X)  #Original Data

    # Variable dimension
    shape = X_copy.shape

    N = shape[0]  # Number of datapoints (time series)
    if N < 2:
        raise ShapeError(
            "At least 2 samples should be given (N>=2)" + str(N)
        )
    # Assume that both d and T are "missing"
    if len(shape) == 1:
        X_copy = np.expand_dims(X_copy, axis=(1,2))
    # Assume that only T is missing
    elif len(shape) == 2:
        X_copy = np.expand_dims(X_copy, axis=1)
    elif len(shape) != 3:
        raise ShapeError(
            f"Invalid shape of datapoints provided: {shape}. "
            + "Please provide a valid shape: "
            + "`(N,)` or `(N, d)` or `(N, T, d)`"
        )
    return X_copy

def set_random_state(
    rng: Union[Generator, int, RandomState],
    rtype: type,
) -> Union[Generator, int, RandomState]:
    """
    Set the random state to a given type.

    Scikit-learn and aeon use np.random.RandomState (or int) while numpy encourages the use of np.random.Generator. In addition, P This function allows to convert

    Parameters
    ----------
    rng : Union[np.random.Generator, int, np.random.RandomState]
        The random state to set. Can be a `numpy.random.Generator`, an
        integer seed, or a `numpy.random.RandomState`.
    rtype : type
        The type of random state to return. Can be a
        `numpy.random.Generator`, an integer seed type, or a
        `numpy.random.RandomState`.

    Returns
    -------
    Union[np.random.Generator, int, np.random.RandomState]
        The random state of the specified type.

    Raises
    ------
    ValueError
        If the provided `rng` is not of a valid type.
    ValueError
        If the provided `rtype` is not of a valid type.
    """
    # Base case: the given type is the correct type, then do nothing
    if isinstance(rng, rtype):
        return rng

    # Otherwise make everything an int first
    # Instanciate a rng if an int was given
    if isinstance(rng, Generator):
        # From Generator to int
        seed = int(rng.integers(10000))
    elif isinstance(rng, RandomState):
        # From RandomState to int
        seed = int(rng.randint(10000))
    elif isinstance(rng, int):
        seed = rng
    else:
        raise ValueError(f"Invalid RandomGenerator type: Got {type(rng)}, expected numpy.random.Generator, or int or np.random.RandomState.")

    # Now return the correct type
    if rtype is int:
        return seed
    elif rtype is Generator:
        return np.random.default_rng(seed)
    elif rtype is RandomState:
        return np.random.RandomState(seed)
    else:
        raise ValueError(f"Invalid RandomGenerator return type: Got {rtype}, expected numpy.random.Generator, or int or np.random.RandomState.")


def _get_model_parameters(
    model_class,
    model_kw: dict = {},
    fit_predict_kw: dict = {},
    model_class_kw: dict = {},
) -> Tuple[Dict, Dict, Dict]:
    """
    Initialize clustering model parameters.

    Parameters
    ----------
    model_class : type
        Clustering model class.
    model_kw : dict, optional
        Model initialization keyword arguments.
    fit_predict_kw : dict, optional
        Keyword arguments for the model `fit_predict` call.
    model_class_kw : dict, optional
        Keyword arguments describing model API specifics.

    Returns
    -------
    Tuple[Dict, Dict, Dict]
        Tuple containing initialization kwargs, fit/predict kwargs,
        and model class metadata kwargs.
    """
    m_kw = {}
    ft_kw = {}
    mc_kw = {
        "k_arg_name" : "n_clusters",
        "X_arg_name" : "X"
    }
    # ----------- method specific key-words ------------------------
    if model_class == KMeans:
        m_kw = {
            'max_iter' : model_kw.pop('max_iter', 100),
            'n_init' : model_kw.pop('n_init', 20),
            'tol' : model_kw.pop('tol', 1e-3),
        }
        ft_kw.update(fit_predict_kw)
    elif model_class == GaussianMixture:
        mc_kw['k_arg_name'] = "n_components"
        mc_kw.update(model_class_kw)
    m_kw.update(model_kw)
    ft_kw.update(fit_predict_kw)
    mc_kw.update(model_class_kw)
    return m_kw, ft_kw, mc_kw

def default_ts_average_kwargs(
    user_kwargs: dict = {},
) -> dict:
    """
    Complete provided kwargs with default ones for time-series average
    functions

    Add the following pairs to the user provided kwargs (whose value)
    will be overriden if a corresponding key-value pair is provided by
    the user:

    ``{ "distance": "msm", "init_barycenter": "medoids", "method":
    "petitjean", "random_state" : 221}``

    These kwargs are going to be used with the
    `aeon.clustering.averaging.elastic_barycenter_average
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.clustering.averaging.elastic_barycenter_average.html#elastic-barycenter-average>`_
    function, and in addition, by default (if ``"distance": "msm"``),
    this function calls `aeon.distances.msm_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.msm_distance.html#aeon.distances.msm_distance>`_.

    Returns
    -------
    dict
        Default time-series kwargs for average functions
    """

    final_kwargs = {
        "distance": "msm",
        "init_barycenter": "medoids",
        "method": "petitjean",
        "random_state" : 221,
    }


    final_kwargs.update(user_kwargs)

    return final_kwargs

def default_ts_distance_kwargs(
    user_kwargs: dict = {},
) -> dict:
    """
    Complete provided kwargs with default ones for time-series distance metrics

    Adds a ``{"method" : "msm"}`` pair to the user provided
    kwargs (whose value will be overriden if a corresponding key-value
    pair is provided by the user).

    If a custom callable is given (using the ``"CALLABLE"`` key) or if
    another method than ``"msm"`` is used, then the default kwargs are
    simply ``{}``, which means only user-specified kwargs are used.

    If no custom callable is given, the function used is
    `aeon.distances.pairwise_distance
    <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.pairwise_distance.html#pairwise-distance>`_

    Returns
    -------
    dict
        Default Time-Series distance kwargs
    """
    if (
        ("CALLABLE" in user_kwargs)
        or ("method" in user_kwargs and user_kwargs["method"] != "msm")
    ):
        final_kwargs = {}
    else:
        final_kwargs = {
            "method" : "msm",
        }

    final_kwargs.update(user_kwargs)

    return final_kwargs
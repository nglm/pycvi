import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import List, Sequence, Union, Any, Dict, Tuple

def set_data_shape(X: np.ndarray) -> np.ndarray:
    """
    Returns a copy of the data but with the right shape
    """
    X_copy = np.copy(X)  #Original Data

    # Variable dimension
    shape = X_copy.shape

    N = shape[0]  # Number of members (time series)
    if N < 2:
        raise ValueError(
            "At least 2 samples should be given (N>=2)" + str(N)
        )
    # Assume that both d and T are "missing"
    if len(shape) == 1:
        X_copy = np.expand_dims(X_copy, axis=(1,2))
    # Assume that only T is missing
    elif len(shape) == 2:
        X_copy = np.expand_dims(X_copy, axis=1)
    elif len(shape) != 3:
        raise ValueError(
            "Invalid shape of members provided: {shape}. "
            + "Please provide a valid shape: (N,) or (N, d) or (N, T, d)"
        )
    return X_copy

def get_model_parameters(
    model_class,
    model_kw: dict = {},
    fit_predict_kw: dict = {},
    model_class_kw: dict = {},
) -> Tuple[Dict, Dict, Dict]:
    """
    Initialize clustering model parameters

    :return: 2 dict, for the model initialization and its fit method
    :rtype: Tuple[Dict, Dict]
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
        m_kw.update(model_kw)
        ft_kw.update(fit_predict_kw)
    elif model_class == GaussianMixture:
        mc_kw['k_arg_name'] = "n_components"
        mc_kw.update(model_class_kw)
    return m_kw, ft_kw, mc_kw
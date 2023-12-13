"""
Generate clusterings and compute clustering-related information.

.. rubric:: Main functions

The main functions of this module are:

- :func:`pycvi.cluster.generate_all_clusterings`, that generate all clusterings for a given range of number of clusters :math:`k`.
- :func:`pycvi.cluster.compute_center`, that computes the center of a cluster.

"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw_path
from tslearn.barycenters import softdtw_barycenter
from typing import List, Sequence, Union, Any, Dict, Tuple
from ._configuration import set_data_shape, get_model_parameters
from .exceptions import ShapeError, NoClusterError

def compute_center(
    cluster: np.ndarray,
) -> np.ndarray:
    """
    Compute the center of a cluster.

    For non time-series data, this is simply the average of all
    datapoints in the given cluster, but for time-series data and when
    DTW is used as the distance measure, then the cluster center is
    defined as the DBA (DTW barycentric average) as defined by Petitjean
    et al [DBA]_.

    Note that the `"N"` dimension is not included in the result.

    .. [DBA] F. Petitjean, A. Ketterlin, and P. Gan carski, “A global
       averaging method for dynamic time warping, with applications to
       clustering,” *Pattern Recognition*, vol. 44, pp. 678–693, Mar.
       2011.

    Parameters
    ----------
    cluster : np.ndarray, shape `(N, d*w_t)` or `(N, w_t, d)`
        Data values in this cluster.

    Returns
    -------
    np.ndarray, shape `(d*w_t)` or `(w_t, d)` if DTW is used.
        The cluster center.
    """
    dims = cluster.shape

    # Regular case
    if len(dims) == 2:
        # Shape (d)
        center = np.mean(cluster, axis=0).reshape(-1)

    # DTW case
    elif len(dims) == 3:
        # Shape (w_t, d)
        center = softdtw_barycenter(cluster)

    else:
        msg = (
            "Clusters should be of dimension (N, d*w_t) or (N, w_t, d)."
            + "and not " + str(dims)
        )
        raise ShapeError(msg)
    return center


def generate_uniform(
    data: np.ndarray,
    zero_type: str = "bounds",
    N_zero: int = 10,
) -> List[np.ndarray]:
    """
    Generate `N_zero` samples from a uniform distribution based on data.

    `data` and each element of the returned `l_data0` have the same
    shape, either `(N, T, d)` or `(N, T*d)` if DTW is used.

    Parameters
    ----------
    data : np.ndarray
        The original dataset
    zero_type : str, optional
        Determines how to parametrize the uniform
        distribution to sample from in the case :math:`k=0`, by default
        "bounds". Possible options:

        - `"variance"`: the uniform distribution is defined such that it
          has the same variance and mean as the original data.
        - `"bounds"`: the uniform distribution is defined such that it
          has the same bounds as the original data.

    N_zero : int, optional
        Number of uniform distributions sampled, by default 10

    Returns
    -------
    List[np.ndarray]
        A list of samples from a uniform distribution, parametrized
        according to the original dataset given `data`
    """
    # Determines how to measure the score of the 0th component
    if zero_type == 'variance':
        # Get the parameters of the uniform distrib using mean and variance
        # mean = (1/2)*(a+b)
        # var  = (1/12)*(b-a)**2
        var = np.var(data, axis=0, keepdims=True)[0]
        mean = np.mean(data, axis=0, keepdims=True)[0]

        mins = (2*mean - np.sqrt(12*var)) / 2
        maxs = (2*mean + np.sqrt(12*var)) / 2
    else:
        # Get the parameters of the uniform distrib using min and max We
        # keep all the dims except the first one (Hence the 0) because
        # The number of members dimension will be added in members_0 in
        # the List comprehension
        # data of shape (N, T, d) even if d and T were initially omitted
        # mins and max of shape (T, d) (the [0] is to get rid of the N
        # dim)
        mins = np.amin(data, axis=0, keepdims=True)[0]
        maxs = np.amax(data, axis=0, keepdims=True)[0]

    # Generate N_zero samples from a uniform distribution with shape
    # the same shape as data
    l_data0 = [
        np.random.uniform(low=mins, high=maxs, size=data.shape)
        for _ in range(N_zero)
    ]
    return l_data0

def prepare_data(
    X: np.ndarray,
    DTW: bool = False,
    window: dict = None,
    transformer: callable = None,
    scaler = StandardScaler(),
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Data to be used for computing clusters and CVIs

    Scaler has to be fit beforehand on the original data (even for the
    case :math:`k=0`).

    X_clus is:

    - a list of :math:`T` `(N, w_t, d)` arrays if sliding window and DTW
      was used
    - a list of :math:`T` `(N, w_t*d)` arrays if sliding window was used
      but not DTW
    - a list of :math:`1` `(N, T, d)` array  if DTW is used but not
      sliding window
    - a list of :math:`1` `(N, T*d)` array if DTW and sliding window
      were not used

    Parameters
    ----------
    X : np.ndarray, shape `(N, T, d)`
        Original data
    window : dict, optional
        Information related to the sliding windows of time-series.

    Returns
    -------
    Union[List[np.ndarray], np.ndarray]
        The processed data, ready to being clustered.
    """

    (N, T, d) = X.shape
    X_clus = []

    if transformer is not None:
        X_trans = transformer(X)
    else:
        X_trans = np.copy(X)

    # Scaling for each variable and not time step wise
    if scaler is not None:
        X_trans = scaler.transform(X_trans.reshape(N*T, d)).reshape(N, T, d)

    # If we use sliding windows, we return a list of extracted windows
    if window is not None:

        for t in range(T):
            ind = window["origin"][t]
            extracted_window = X_trans[:,ind,:]
            if DTW:
                # List of T (N, w_t, d) arrays
                X_clus.append(extracted_window)
            else:
                # List of T (N, w_t*d) arrays
                X_clus.append(extracted_window.reshape(N, -1))
    # Otherwise we return an array
    else:
        if DTW:
            # List of one array of shape (N, T, d)
            X_clus = [X_trans]
        else:
            # List of one array of shape (N, T*d)
            X_clus = [X_trans.reshape(N, -1)]

    # X_clus is:
    # - a list of T (N, w_t, d) arrays if sliding window and DTW was used
    # - a list of T (N, w_t*d) arrays if sliding window was used but not DTW
    # - a list of 1 (N, T, d) array  if DTW is used but not sliding window
    # - a list of 1 (N, T*d) array if DTW and sliding window were not used
    return X_clus

def sliding_window(T: int, w: int) -> dict:
    """
    Compute information related to the sliding windows of time-series.

    Assuming that we consider an array of length :math:`T`, and with indices
    :math:`[0 \\cdots T-1]`.

    Windows extracted are shorter when considering the beginning and the
    end of the array. Which means that a padding is implicitly included.

    When the time window :math:`w` is an *even* number, favor future
    time steps, i.e., when extracting a time window around the datapoint
    :math:`t`, the time window indices are :math:`[t - (w-1)//2, ...,
    t, ..., t + w/2]`.
    When `w` is odd then the window is :math:`[t - (w-1)/2 \\cdots t
    \\cdots t + (w-1)/2]`.

    Which means that the padding is as follows:

    - beginning: `(w-1)//2
    - end: `w//2`

    And that the original indices are as follows:

    - :math:`[0, ..., t + w//2]`, until :math:`t = (w-1)//2`
    - :math:`[t - (w-1)//2, ..., t, ..., t + w//2]` for datapoints in
      :math:`[(w-1)//2, ..., T-1 - w//2]`
    - :math:`[t - (w-1)//2, ..., T-1]` from :math:`t = T-1 - w//2`
    - Note that for :math:`t = (w-1)//2` or :math:`t = (T-1 - w//2)`,
      both formulas apply.
    - Note also that the left side of the end case is the same as the
      left side of the base case

    Window sizes:

    - :math:`(1 + w//2)` at :math:`t=0`, then :math:`t + (1 + w//2)`
      until :math:`t = (w-1)//2`
    - All datapoints from :math:`[(w-1)//2, ..., T-1 - w//2]` have a
      normal window size.
    - :math:`(w+1)//2` at :math:`t=T-1`, then :math:`T-1-t + (w+1)//2`
      from :math:`t = T-1 - w//2`
    - Note that for :math:`t = (w-1)//2` or :math:`t = (T-1 - w//2)`,
      both formulas apply

    Consider an extracted time window of length `w_real` (with
    :math:`w\_real \\leq w`, if the window was extracted at the
    beginning or the end of the array). The midpoint of the extracted
    window (i.e. the index in that window that corresponds to the
    datapoint around which the time window was extracted in the original
    array) is:

    - :math:`0` at :math:`t=0`, then :math:`t`, until :math:`t =
      pad\_left`, i.e. :math:`t = (w-1)//2`
    - For all datapoints between, :math:`[(w-1)//2, ..., (T-1 - w//2)]`,
      the midpoint is :math:`(w-1)//2` (so it is the same as the base
      case)
    - :math:`w\_real-1` at :math:`t=T-1`, then :math:`w\_real - (T-t)`,
      from :math:`t=T-1-pad\_right`, i.e. from :math:`t = (T-1 - w//2)`
    - Note that for :math:`t = (w-1)//2` or :math:`t = (T-1 - w//2)`,
      both formulas apply.

    The midpoint in the original array is actually simply :math:`t`.

    Available keys:

    - `"padding_left"`: Padding to the left
    - `"padding_right"`: Padding to the right
    - `"length"`: Actual length of the time window
    - `"midpoint_w"`: Midpoint in the window reference
    - `"midpoint_o"`: Midpoint in the origin reference
    - `"origin"`: Original indices

    """
    # Boundaries between regular cases and extreme ones
    ind_start = (w-1)//2
    ind_end = T-1 - w//2
    pad_l = (w-1)//2
    pad_r = (w//2)
    window = {}
    window["padding_left"] = pad_l
    window["padding_right"] = pad_r

    # For each t in [0, ..., T-1]...
    # ------- Real length of the time window -------
    window["length"] = [w for t in range(T)]
    window["length"][:ind_start] = [
        t + (1 + w//2)
        for t in range(0, ind_start)
    ]
    window["length"][ind_end:] = [
        T-1-t + (w+1)//2
        for t in range(ind_end, T)
    ]
    # ------- Midpoint in the window reference -------
    # Note that the end case is the same as the base case
    window["midpoint_w"] = [(w-1)//2 for t in range(T)]
    window["midpoint_w"][:ind_start]  = [ t for t in range(0, ind_start) ]
    # ------- Midpoint in the origin reference -------
    window["midpoint_o"] = [t for t in range(T)]
    # ------- Original indices -------
    window["origin"] = [
        # add a +1 to the range to include last original index
        list(range( (t - (w-1)//2),  (t + w//2) + 1))
        for t in range(T)
    ]
    window["origin"][:ind_start]  = [
        # add a +1 to the range to include last original index
        list(range(0, (t + w//2) + 1))
        for t in range(0, ind_start)
    ]
    window["origin"][ind_end:] = [
        # Note that the left side is the same as the base case
        list(range( (t - (w-1)//2),  (T-1) + 1 ))
        for t in range(ind_end, T)
    ]
    return window

def _get_clusters(
    model,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    model_class_kw : Dict = {},
) -> List[List[int]]:
    """
    Generate a clustering instance with the given model/fit parameters

    Parameters
    ----------
    fit_predict_kw : dict, optional
        Dict of kw for the fit_predict method, defaults to {}
    model_class_kw : dict, optional
        To know how X and n_clusters args are called in this model
        class, defaults to {}

    Returns
    -------
    List[List[int]]
        Members affiliation to the generated clustering
    """
    n_clusters = model_kw[model_class_kw.get("k_arg_name", "n_clusters")]
    X = fit_predict_kw[model_class_kw.get("X_arg_name", "X")]
    N = len(X)

    # ====================== Fit & predict part =======================
    labels = model(**model_kw).fit_predict(**fit_predict_kw)

    # ==================== clusters ======================
    clusters = [ [] for _ in range(n_clusters)]
    for label_i in range(n_clusters):
        # Members belonging to that clusters
        members = [m for m in range(N) if labels[m] == label_i]
        clusters[label_i] = members
        if members == []:
            raise NoClusterError('No members in cluster')

    return clusters

def generate_all_clusterings(
    data: np.ndarray,
    model_class,
    n_clusters_range: Sequence = None,
    DTW: bool = True,
    time_window: int = None,
    transformer: callable = None,
    scaler = StandardScaler(),
    model_kw: dict = {},
    fit_predict_kw: dict = {},
    model_class_kw: dict = {},
    quiet: bool = True,
) -> List[Dict[int, List[List[int]]]]:
    """
    Generate and return all clusterings.

    ```clusterings_t_k[t_w][k][i]``` is a list of members indices
    contained in cluster :math:`i` for the clustering that assumes
    :math:`k` clusters for the extracted time window :math:`t\_w`.

    If some clusterings couldn't be defined because the clustering
    algorithm didn't converged then ```clusterings_t_k[t_w][n_clusters]
    = None```.

    Parameters
    ----------
    data : np.ndarray,
        Original data. Acceptable input shapes and their corresponding
        output shapes in the PyCVI package:

        - `(N,)` -> `(N, 1, 1)`
        - `(N, d)` -> `(N, 1, d)`
        - `(N, T, d)` -> `(N, T, d)`
    model_class : A sklearn-like clustering class
        A class implementing a clustering algorithm.
    n_clusters_range : Sequence, optional
        Assumptions on the number of clusters to try out, by default
        None. If None, `n_clusters_range=range(N+1)`.
    DTW : bool, optional
        Determines if DTW should be used as the distance measure
        (concerns only time series data), by default True.
    time_window : int, optional
        Length of the sliding window (concerns only time-series data),
        by default None. If None, no sliding window is used, and the
        time series is considered as a whole.
    transformer : callable, optional
        A potential additional preprocessing step, by default None. If
        None, no transformation is applied on the data
    scaler : A sklearn-like scaler model, optional
        A data scaler, by default StandardScaler(). If None, so scaling
        is applied on the data.
    model_kw : dict, optional
        Specific kwargs to give to `model_class` init method, by default
        {}
    fit_predict_kw : dict, optional
        Specific kwargs to give to the `fit_predict` method of the
        `model_class` clustering model, by default {}.
    model_class_kw : dict, optional
        Dictionary that contains the argument names of the number of
        clusters and the data to give to the clustering model, by
        default {}, which then updated as follows: `{"k_arg_name" :
        "n_clusters", "X_arg_name" : "X" }` to follow sklean
        conventions.
    quiet : bool, optional
        Controls the verbosity of the function, by default True.

    Returns
    -------
    List[Dict[int, List[List[int]]]]
        All clusterings for the given range on the number of clusters
        and for the potential sliding windows if applicable.
    """
    # --------------------------------------------------------------
    # --------------------- Preliminary ----------------------------
    # --------------------------------------------------------------

    data_copy = set_data_shape(data)
    (N, T, d) = data_copy.shape
    if scaler is not None:
        scaler.fit(data_copy.reshape(N*T, d))

    if n_clusters_range is None:
        n_clusters_range = range(N+1)

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
    n_windows = len(data_clus)

    # --------------------------------------------------------------
    # --------------------- Find clusters --------------------------
    # --------------------------------------------------------------

    # clusterings_t_k[t_w][k][i] is a list of members indices contained
    # in cluster i for the clustering assuming k clusters for the
    # extracted time window t_w
    clusterings_t_k = [{} for _ in range(n_windows)]

    for t_w in range(n_windows):
        # ---- clustering method specific parameters --------
        # X_clus of shape (N, w_t*d) or (N, w_t, d)
        X_clus = data_clus[t_w]

        # Get clustering model parameters required by the
        # clustering model
        model_kw, fit_predict_kw, model_class_kw = get_model_parameters(
            model_class,
            model_kw = model_kw,
            fit_predict_kw = fit_predict_kw,
            model_class_kw = model_class_kw,
        )
        # Update fit_predict_kw with the data
        fit_predict_kw[model_class_kw["X_arg_name"]] = X_clus

        for n_clusters in n_clusters_range:

            # All members in the same cluster. Go to next iteration
            if n_clusters <= 1:
                clusterings_t_k[t_w][n_clusters] = [[i for i in range(N)]]
            # General case
            else:

                # Update model_kw with the number of clusters
                model_kw[model_class_kw["k_arg_name"]] = n_clusters

                # ---------- Fit & predict using clustering model-------
                try :
                    clusters = _get_clusters(
                        model_class,
                        model_kw = model_kw,
                        fit_predict_kw = fit_predict_kw,
                        model_class_kw = model_class_kw,
                    )
                except NoClusterError as e:
                    if not quiet:
                        print(str(e))
                    clusters = None

                clusterings_t_k[t_w][n_clusters] = clusters

    return clusterings_t_k
import numpy as np
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw_path
from tslearn.barycenters import softdtw_barycenter
from typing import List, Sequence, Union, Any, Dict


def compute_asym_std(X, center):
    # How many members above/below the average
    # "+1" because otherwise the std goes up to the member if there is just
    # one member
    n = len(X)+1
    X = np.array(X)
    return np.sqrt( np.sum((X - center)**2) / n)

def compute_disp(X, center):
    (d, ) = center.shape                      # shape: (d)
    disp_inf = np.zeros(d)                  # shape: (d)
    disp_sup = np.zeros(d)                  # shape: (d)
    disp = np.std(X, axis=0).reshape(-1)    # shape: (d)
    for i in range(d):
        # Get the members below/above the average
        X_inf = [m for m in X[:, i] if m <= center[i]]
        X_sup = [m for m in X[:, i] if m >= center[i]]

        disp_inf[i] = compute_asym_std(X_inf, center[i])
        disp_sup[i] = compute_asym_std(X_sup, center[i])

    return (disp, disp_inf, disp_sup)

def compute_center(
    cluster: np.ndarray,
):
    """
    Compute the center of a cluster (mean or barycenter if using DTW)
    The "n_samples" dimension is not included in the result.
    """
    dims = cluster.shape

    # Regular case
    if len(dims) == 2:
        # Shape (d)
        return np.mean(cluster, axis=0).reshape(-1)

    # DTW case
    elif len(dims) == 3:
        # Shape (w_t, d)
        return softdtw_barycenter(cluster)

    else:
        msg = (
            "Clusters should be of dimension (N, d*w_t) or (N, w_t, d)."
            + "and not " + str(dims)
        )
        raise ValueError(msg)

def align_to_barycenter(
    cluster,
    barycenter,
    midpoint_w,
):
    aligned_X = []
    # For each time series in the dataset, compare to the barycenter
    for ts in cluster:
        path, _ = dtw_path(
            barycenter, ts,
            global_constraint="sakoe_chiba", sakoe_chiba_radius=5
        )
        # Find time steps that match with the midpoint of the
        # barycenter
        ind = [
            path[i][1] for i in range(len(path))
            if path[i][0] == midpoint_w
        ]
        # Option 1: take all time steps that match the midpoint of
        # the barycenter
        # Here we can have len(X) > N_clus! because for a given time
        # series, multiple time step can correspond to the midpoint
        # of the barycenter find finding the dtw_path between that
        # time series and this cluster. Each element of X is of
        # shape (d)
        # X += [ts[i] for i in ind]

        # Option 2: take all the mean value for all time steps that
        # match the midpoint of the barycenter
        aligned_X.append(np.mean([ts[i] for i in ind], axis=0))
    aligned_X = np.array(aligned_X)
    return aligned_X

def compute_cluster_params(
    cluster: np.ndarray,
    midpoint_w: int = None,
) -> Dict:
    """
    Compute the mean, std, std_sup, inf, etc of a given cluster

    If `cluster` has shape `(N_clus, w_t, d)`, uses DTW. Otherwise, it
    must have a shape `(N_clus, w_t*d)` and the regular mean is used.

    If DTW is used, first compute the barycentric average of the
    cluster. Then consider only the midpoint of that barycenter to
    compute the mean and uses it as a reference to compute the standard
    deviation. Otherwise, uses the regular mean.

    :param cluster: Values of the members belonging to that cluster
    :type cluster: np.ndarray, shape (N_clus, w_t*d)`
    or `(N_clus, w_t, d)`
    :param midpoint_w: center point of the time window w_t, used as
    reference in case of DTW, defaults to None Note that for the first
    time steps and the last time steps, the 'center' of the time window
    is not necessarily in the middle on the window. E.g. for t=0 and w =
    50, we have w_t = 26 and midpoint_w = 0
    :type midpoint_w: int, optional
    :return: Dict of summary statistics
    :rtype: Dict
    """
    dims = cluster.shape
    cluster_params = {}

    # Regular case
    if len(dims) == 2:
        # Compute center
        center = compute_center(cluster)
        X = np.copy(cluster)

    # DTW case
    else:
        (N_clus, w_t, d) = dims
        if midpoint_w is None:
            midpoint_w = w_t // 2
        # Take the barycenter as reference
        barycenter = compute_center(cluster)
        center = barycenter[midpoint_w]
        cluster_params['barycenter'] = barycenter

        # Align to barycenter
        X = align_to_barycenter(cluster, barycenter, midpoint_w)

    # Compute dispersion
    (disp, disp_inf, disp_sup) = compute_disp(X, center)

    # Member values (aligned with the barycenter if DTW was used)
    cluster_params['X'] = X
    cluster_params['center'] = center
    cluster_params['disp'] = disp
    cluster_params['disp_inf'] = disp_inf
    cluster_params['disp_sup'] = disp_sup
    return cluster_params

def generate_uniform(
    X: np.ndarray,
    zero_type: str = "bounds",
) -> np.ndarray:
    """
    Generate a uniform distribution with the same bounds as X

    X and X0 of shape (N, d, T)
    """
    # Determines how to measure the score of the 0th component
    if zero_type == 'variance':
        raise NotImplementedError("Only 'bounds' is implemented as `zero_type`")
        # Get the parameters of the uniform distrib using mean and variance
        var = np.var(X, axis=0)
        mean = np.mean(X, axis=0)

        mins = (2*mean - np.sqrt(12*var)) / 2
        maxs = (2*mean + np.sqrt(12*var)) / 2

    # Get the parameters of the uniform distrib using min and max
    # We keep all the dims except the first one (Hence the 0) because
    # The number of members dimension will be added in members_0 in the
    # List comprehension
    # X of shape (N, d, T) even if d and T were initially omitted
    # mins and max of shape (d, T) (the [0] is to get rid of the N dim)
    mins = np.amin(X, axis=0, keepdims=True)[0]
    maxs = np.amax(X, axis=0, keepdims=True)[0]

    # Generate a perfect uniform distribution
    N = len(X)
    steps = (maxs-mins) / (N-1)
    X0 = np.array([mins + i*steps for i in range(N)])
    return X0

def prepare_data(
    X: np.ndarray,
    DTW: bool = False,
    window: dict = None,
    transformer: callable = None,
    scaler = StandardScaler(),
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Data to be used for cluster, scores and params

    The index that should be used to compute cluster params
    of clustering that were computed using `X_clus` is called
    "midpoint_w" in the window dictionary

    X_clus is:
    - a list of T (N, w_t, d) arrays if sliding window and DTW was used
    - a list of T (N, w_t*d) arrays if sliding window was used but not DTW
    - a list of 1 (N, T, d) array  if DTW is used but not sliding window
    - a list of 1 (N, T*d) array if DTW and sliding window were not used

    :param X: data of shape (N, T, d)
    :type X: np.ndarray
    :param window: _description_
    :type window: dict
    :return: The prepared data
    :rtype: Union[List[np.ndarray], np.ndarray]
    """

    (N, T, d) = X.shape
    X_clus = []

    if transformer is not None:
        X_trans = transformer(X)
    else:
        X_trans = np.copy(X)

    # Scaling for each variable and not time step wise
    if scaler is not None:
        X_trans = scaler.fit_transform(X_trans.reshape(N, -1)).reshape(N, T, d)

    # If we use sliding windows, we return a list of extracted windows
    if window is not None:

        for t in range(T):
            ind = window["origin"][t]
            extracted_window = X_trans[:,ind,:]
            if DTW:
                # List of (N, w_t, d) arrays
                X_clus.append(extracted_window)
            else:
                # List of (N, w_t*d) arrays
                X_clus.append(extracted_window.reshape(N, -1))
    # Otherwise we return an array
    else:
        if DTW:
            # array of shape (N, T, d)
            X_clus = [X_trans]
        else:
            # array of shape (N, T*d)
            X_clus = [X_trans.reshape(N, -1)]

    # X_clus is:
    # - a list of T (N, w_t, d) arrays if sliding window and DTW was used
    # - a list of T (N, w_t*d) arrays if sliding window was used but not DTW
    # - a list of 1 (N, T, d) array  if DTW is used but not sliding window
    # - a list of 1 (N, T*d) array if DTW and sliding window were not used
    return X_clus

def sliding_window(T: int, w: int) -> dict:
    """
    Assuming that we consider an array of length `T`, and with indices
    `[0, ..., T-1]`.

    Windows extracted are shorter when considering the beginning and the
    end of the array. Which means that a padding is implicitly included.

    When the time window `w` is an *even* number, favor future time steps, i.e.,
    when extracting a time window around the datapoint t, the time window
    indices are [t - (w-1)//2, ... t, ..., t + w/2].
    When `w` is odd then the window is [t - (w-1)/2, ... t, ..., t + (w-1)/2]

    Which means that the padding is as follows:

    - beginning: (w-1)//2
    - end: w//2

    And that the original indices are as follows:

    - [0, ..., t + w//2], until t = (w-1)//2
    - [t - (w-1)//2, ..., t, ..., t + w//2] for datapoints in
    [(w-1)//2, ..., T-1 - w//2]
    - [t - (w-1)//2, ..., T-1] from t = T-1 - w//2
    - Note that for t = (w-1)//2 or t = (T-1 - w//2), both formulas apply.
    - Note also that the left side of the end case is the same as the left
    side of the base case

    Window sizes:

    - (1 + w//2) at t=0, then t + (1 + w//2) until t = (w-1)//2
    - All datapoints from [(w-1)//2, ..., T-1 - w//2] have a normal
    window size.
    - (w+1)//2 at t=T-1, then T-1-t + (w+1)//2 from t = T-1 - w//2
    - Note that for t = (w-1)//2 or t = (T-1 - w//2), both formulas apply

    Consider an extracted time window of length w_real (with w_real <= w,
    if the window was extracted at the beginning or the end of the array).
    The midpoint of the extracted window (i.e. the index in that window
    that corresponds to the datapoint around which the time window was
    extracted in the original array is:

    - 0 at t=0, then t, until t = pad_left, i.e. t = (w-1)//2
    - For all datapoints between, [(w-1)//2, ..., (T-1 - w//2)], the
    midpoint is (w-1)//2 (so it is the same as the base case)
    - w_real-1 at t=T-1, then w_real - (T-t), from t=T-1-pad_right, i.e.
    from t = (T-1 - w//2)
    - Note that for t = (w-1)//2 or t = (T-1 - w//2), both formulas apply.

    The midpoint in the original array is actually simply t
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

def get_clusters(
    model,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    model_class_kw : Dict = {},
) -> List[List[int]]:
    """
    Generate a clustering instance with the given model/fit parameters

    :param fit_predict_kw: Dict of kw for the fit_predict method,
    defaults to {}
    :type fit_predict_kw: dict, optional
    :param model_class_kw: To know how X and n_clusters args are called in this
    model class, defaults to {}
    :type model_class_kw: dict, optional
    :return: Members affiliation to the generated clustering
    :rtype: List[List[int]]
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
            raise ValueError('No members in cluster')

    return clusters
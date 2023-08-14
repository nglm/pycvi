import numpy as np
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
    Compute the center of a cluster depending on whether DTW is used.
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
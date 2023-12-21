import numpy as np
from typing import List, Sequence, Union, Any, Dict, Tuple

datapoints_with_dup_equal_dist = np.array([
    (0.1 ,   1.,    2.,   1.,     0.1 ),
    (0.05,   0,     0,    0,      0.005 ),
    (0.1,   -1,    -2,   -1,      0.05 ),
    (0.05,  -0.4,  -0.6, -0.5,   -1),
])

datapoints_with_equal_dist = np.array([
    (0.11,   1.,    2.,   1.,     0.1 ),
    (0.05,   0,     0,    0,      0.005 ),
    (0.1,   -1,    -2,   -1,      0.05 ),
    (0.005, -0.4,  -0.6, -0.5,   -1),
])

datapoints = np.array([
    (0.11,   1.1,   2.1,  1.1,    0.1),
    (0.05,   0,     0,    0,      0.005),
    (0.1,   -1,    -2,   -1,      0.05),
    (0.005, -0.4,  -0.6, -0.55,  -1),
])

datapoints_bis = np.array([
    ( 0.2 ,  0.3,    1,      1.1,    0.6),
    ( 0.1,   0.15,   0.5,    0.6,    0.4),
    (-0.1,  -0.2,   -1,     -1.2,   -1 ),
    (-0.3,  -0.4,   -0.6,   -0.7,   -1),
])

datapoints_biv = np.ones((4, 5, 2))
datapoints_biv[:, :, 0] = datapoints
datapoints_biv[:, :, 1] = datapoints_bis

def mini(
    multivariate: bool = False,
    as_time_series: bool = True,
    time_scale: bool = True,
    duplicates: bool = False,
    equal_dist: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a mini dataset for testing purpose

    :param multivariate: Use multivariate data, defaults to False
    :type multivariate: bool, optional
    :param as_time_series: Treat each time step separately if True,
        otherwise, consider time steps as different variable, defaults
        to True
    :type as_time_series: bool, optional
    :param time_scale: get a time axis different from the indices,
    defaults to True
    :type time_scale: bool, optional
    :param duplicates: Allow for duplicates for some time steps,
    defaults to False
    :type duplicates: bool, optional
    :param equal_dist: Allow for datapoint being at equal distance from
        2 other datapoints, defaults to False
    :type equal_dist: bool, optional
    :return: _description_
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    (N, T) = datapoints.shape
    time = np.arange(T)

    # Get a time axis different from the indices (by a factor 6)
    if time_scale:
        time *= 6
    # Bivariate data
    if multivariate:
        data = np.ones((N, T, 2))
        data[:, :, 0] = datapoints
        data[:, :, 1] = datapoints_bis
        data = datapoints_biv
    # Univariate data
    else:
        # Add duplicate values at some time steps
        if duplicates:
            data = datapoints_with_dup_equal_dist
        else:
            # Add datapoints
            if equal_dist:
                data = datapoints_with_equal_dist
            else:
                data = datapoints
        # Add the "d" dimension with d=1
        data = np.expand_dims(data, -1)
    # Treat time series as multivariate data instead of time series
    # New shape: (N, 1, d*T)
    if not as_time_series:
        data = np.reshape(N, 1, -1)
    return np.copy(data), np.copy(time)


def normal(
    k: int = 2,
    nis: List[int] = [20, 10],
    T: int = 5,
    multivariate: bool = True,
    time_scale: bool = True,
    lows = [-2., -1.5],
    highs = [1., 3.],
    sigmas = [0.5, 1.],
) -> Tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(221)

    time = np.arange(T)
    # Get a time axis different from the indices (by a factor 6)
    if time_scale:
        time *= 6

    if not multivariate:
        d = 1
        lows = lows[0]
        highs = highs[0]
        sigmas = sigmas[0]
    else:
        d = len(lows)

    mus = [rng.uniform(lows, highs) for _ in range(k)]
    stds = [sigmas for _ in range(k)]

    data = np.array([
        rng.normal(mus[i_k], stds[i_k], size=(T, d))
        for i_k, ni in enumerate(nis) for _ in range(ni)
    ])
    return data, time

def get_clusterings(N: int) -> List[List[int]]:
    C = {}
    C["C1"] = [[i for i in range(N)]]
    C["C2"] = [
        [i for i in range(N//2)],
        [i for i in range(N//2, N)],
    ]
    C["C2_bis"] = [
        [i for i in range(N//2-3)],
        [i for i in range(N//2-3, N)],
    ]
    C["C2_inv"] = [
        [i+(N//2) for i in range(N//2)],
        [i-(N//2) for i in range(N//2, N)],
    ]
    C["C2_shuffled"] = [
        C["C2"][1], C["C2"][0]
    ]
    C["C2_bis_shuffled"] = [
        C["C2_bis"][1], C["C2_bis"][0]
    ]
    C["C3"] = [
        [i for i in range(N//3)],
        [i for i in range(N//3, 2*N//3)],
        [i for i in range(2*N//3, N)],
    ]
    C["C3_bis"] = [
        [i for i in range(N//3-5)],
        [i for i in range(N//3-5, 2*N//3)],
        [i for i in range(2*N//3, N)],
    ]
    C["C3_shuffled"] = [
        C["C3"][0], C["C3"][2], C["C3"][1]
    ]
    C["C3_bis_shuffled"] = [
        C["C3_bis"][0], C["C3_bis"][2], C["C3_bis"][1]
    ]
    C["CN"] = [[i] for i in range(N)]
    return C

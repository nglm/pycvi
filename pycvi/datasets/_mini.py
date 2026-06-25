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

# Tiny clustered time-series datasets for testing.
# Cluster A (first 3 samples): increasing profiles.
# Cluster B (last 3 samples): decreasing profiles.
datapoints_2 = np.array([
    (0.10, 0.55, 1.00, 1.45, 1.90, 2.25),
    (0.15, 0.60, 1.05, 1.50, 1.95, 2.30),
    (0.05, 0.50, 0.95, 1.40, 1.85, 2.20),
    (2.30, 1.90, 1.45, 1.00, 0.60, 0.20),
    (2.20, 1.80, 1.35, 0.90, 0.50, 0.10),
    (2.40, 2.00, 1.55, 1.10, 0.70, 0.30),
])

datapoints_2_with_dup = np.array([
    (0.10, 0.55, 1.00, 1.00, 1.90, 2.25),
    (0.15, 0.60, 1.05, 1.05, 1.95, 2.30),
    (0.05, 0.50, 0.95, 0.95, 1.85, 2.20),
    (2.30, 1.90, 1.45, 1.45, 0.60, 0.20),
    (2.20, 1.80, 1.35, 1.35, 0.50, 0.10),
    (2.40, 2.00, 1.55, 1.55, 0.70, 0.30),
])

datapoints_2_equal_dist = np.array([
    (0.10, 0.55, 1.00, 1.45, 1.90, 2.25),
    (0.20, 0.65, 1.10, 1.55, 2.00, 2.35),
    (0.00, 0.45, 0.90, 1.35, 1.80, 2.15),
    (2.25, 1.85, 1.40, 0.95, 0.55, 0.15),
    (2.15, 1.75, 1.30, 0.85, 0.45, 0.05),
    (2.35, 1.95, 1.50, 1.05, 0.65, 0.25),
])

datapoints_2_bis = np.array([
    (1.90, 1.55, 1.20, 0.90, 0.65, 0.45),
    (2.00, 1.65, 1.30, 1.00, 0.75, 0.55),
    (1.80, 1.45, 1.10, 0.80, 0.55, 0.35),
    (0.40, 0.65, 0.95, 1.25, 1.60, 1.95),
    (0.30, 0.55, 0.85, 1.15, 1.50, 1.85),
    (0.50, 0.75, 1.05, 1.35, 1.70, 2.05),
])

datapoints_2_biv = np.ones((6, 6, 2))
datapoints_2_biv[:, :, 0] = datapoints_2
datapoints_2_biv[:, :, 1] = datapoints_2_bis

def mini(
    multivariate: bool = False,
    as_time_series: bool = True,
    time_scale: bool = True,
    duplicates: bool = False,
    equal_dist: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a mini dataset for testing purposes.

    Parameters
    ----------
    multivariate : bool, optional
        If True, return multivariate data.
    as_time_series : bool, optional
        If True, treat each time step as temporal structure; otherwise,
        treat time steps as separate variables.
    time_scale : bool, optional
        If True, return a time axis scaled from raw indices.
    duplicates : bool, optional
        If True, include duplicate values at some time steps.
    equal_dist : bool, optional
        If True, include datapoints at equal distance from two others.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing generated data and corresponding time axis.
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


def mini_2(
    multivariate: bool = False,
    time_scale: bool = True,
    duplicates: bool = False,
    equal_dist: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a tiny time-series dataset for testing purposes.

    The returned dataset contains two obvious clusters in time-series
    space: increasing profiles and decreasing profiles.

    Parameters
    ----------
    multivariate : bool, optional
        If True, return multivariate data.
    time_scale : bool, optional
        If True, return a time axis scaled from raw indices.
    duplicates : bool, optional
        If True, include duplicate values at some time steps.
    equal_dist : bool, optional
        If True, include datapoints at equal distance from two others.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing generated time-series data and corresponding
        time axis.
    """
    (N, T) = datapoints_2.shape
    time = np.arange(T)

    if time_scale:
        time *= 6

    if multivariate:
        data = np.copy(datapoints_2_biv)
    else:
        if duplicates:
            data = np.copy(datapoints_2_with_dup)
        elif equal_dist:
            data = np.copy(datapoints_2_equal_dist)
        else:
            data = np.copy(datapoints_2)
        data = np.expand_dims(data, -1)

    return data, np.copy(time)


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

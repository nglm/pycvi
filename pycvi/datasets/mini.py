import numpy as np
from typing import List, Sequence, Union, Any, Dict, Tuple

members_with_dup_equal_dist = np.array([
    (0.1 ,   1.,    2.,   1.,     0.1 ),
    (0.05,   0,     0,    0,      0.005 ),
    (0.1,   -1,    -2,   -1,      0.05 ),
    (0.05,  -0.4,  -0.6, -0.5,   -1),
])

members_with_equal_dist = np.array([
    (0.11,   1.,    2.,   1.,     0.1 ),
    (0.05,   0,     0,    0,      0.005 ),
    (0.1,   -1,    -2,   -1,      0.05 ),
    (0.005, -0.4,  -0.6, -0.5,   -1),
])

members = np.array([
    (0.11,   1.1,   2.1,  1.1,    0.1),
    (0.05,   0,     0,    0,      0.005),
    (0.1,   -1,    -2,   -1,      0.05),
    (0.005, -0.4,  -0.6, -0.55,  -1),
])

members_bis = np.array([
    ( 0.2 ,  0.3,    1,      1.1,    0.6),
    ( 0.1,   0.15,   0.5,    0.6,    0.4),
    (-0.1,  -0.2,   -1,     -1.2,   -1 ),
    (-0.3,  -0.4,   -0.6,   -0.7,   -1),
])

members_biv = np.ones((4, 5, 2))
members_biv[:, :, 0] = members
members_biv[:, :, 1] = members_bis

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
    :param as_time_series: Treat each time step separately if True, otherwise,
    consider time steps as different variable, defaults to True
    :type as_time_series: bool, optional
    :param time_scale: get a time axis different from the indices,
    defaults to True
    :type time_scale: bool, optional
    :param duplicates: Allow for duplicates for some time steps,
    defaults to False
    :type duplicates: bool, optional
    :param equal_dist: Allow for member being at equal distance from 2 other
    members, defaults to False
    :type equal_dist: bool, optional
    :return: _description_
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    (N, T) = members.shape
    time = np.arange(T)

    # Get a time axis different from the indices (by a factor 6)
    if time_scale:
        time *= 6
    # Bivariate data
    if multivariate:
        data = np.ones((N, T, 2))
        data[:, :, 0] = members
        data[:, :, 1] = members_bis
        data = members_biv
    # Univariate data
    else:
        # Add duplicate values at some time steps
        if duplicates:
            data = members_with_dup_equal_dist
        else:
            # Add members
            if equal_dist:
                data = members_with_equal_dist
            else:
                data = members
        # Add the "d" dimension with d=1
        data = np.expand_dims(data, -1)
    # Treat time series as multivariate data instead of time series
    # New shape: (N, 1, d*T)
    if not as_time_series:
        data = np.reshape(N, 1, -1)
    return np.copy(data), np.copy(time)



import numpy as np
from numpy.testing import assert_array_equal
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..datasets import mini
from ..cluster import (
    sliding_window, prepare_data, get_clusters, generate_all_clusterings
)

def test_sliding_window():
    # From examples drawn by hand
    windows = list(range(1,7))
    T = 7
    p_r =     [0, 1, 1, 2, 2, 3]
    p_l =     [0, 0, 1, 1, 2, 2]
    lengths = [
        [1, 1, 1, 1, 1, 1, 1], # 1
        [2, 2, 2, 2, 2, 2, 1], # 2
        [2, 3, 3, 3, 3, 3, 2], # 3
        [3, 4, 4, 4, 4, 3, 2], # 4
        [3, 4, 5, 5, 5, 4, 3], # 5
        [4, 5, 6, 6, 5, 4, 3], # 6
    ]
    midpoints = [
        [0, 0, 0, 0, 0, 0, 0], # 1
        [0, 0, 0, 0, 0, 0, 0], # 2
        [0, 1, 1, 1, 1, 1, 1], # 3
        [0, 1, 1, 1, 1, 1, 1], # 4
        [0, 1, 2, 2, 2, 2, 2], # 5
        [0, 1, 2, 2, 2, 2, 2], # 6
    ]
    origins = [
        [[0], [1], [2], [3], [4], [5], [6]], # 1
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6]], # 2
        [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6]], # 3
        [[0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6], [5, 6]], # 4
        [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6]], # 5
        [[0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6]], # 6
    ]
    for i, w in enumerate(windows):
        window = sliding_window(T, w)

        output = [window["padding_left"], window["padding_right"]]
        output_exp = [p_l[i], p_r[i]]

        output_np = [
            window["length"], window["midpoint_w"],
        ]
        output_np += window["origin"]
        output_np_exp = [lengths[i], midpoints[i]]
        output_np_exp += origins[i]

        for out, out_exp in zip(output, output_exp):
            assert out == out_exp , "out: " + str(out) + " expected " + str(out_exp)
        for out, out_exp in zip(output_np, output_np_exp):
            assert_array_equal(out, out_exp)

def test_prepare_data():
    """
    Test shapes of the output
    """
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        l_w = [1, T//2, T]

        # Using DTW and window
        # data_clus is a list of T (N, w_t, d) arrays
        for w in l_w:
            window = sliding_window(T, w)
            data_clus = prepare_data(
                data,
                DTW=True, window=window, transformer=None,
                scaler=StandardScaler(),
            )

            exp_len = T
            exp_shape = [
                (N, window["length"][t], d)
                for t in range(T)
            ]
            assert (len(data_clus) == exp_len)
            for t in range(T):
                assert data_clus[t].shape == exp_shape[t]

        # Using DTW but not window
        # data_clus is a list of 1 (N, T, d) array
        data_clus = prepare_data(
            data,
            DTW=True, window=None, transformer=None,
            scaler=StandardScaler(),
        )
        assert len(data_clus) == 1
        assert data_clus[0].shape == (N, T, d)

        # Not using DTW but using window
        # data_clus is a list of T (N, w_t*d) arrays
        for w in l_w:
            window = sliding_window(T, w)
            data_clus = prepare_data(
                data,
                DTW=False, window=window, transformer=None,
                scaler=None,
            )

            exp_len = T
            exp_shape = [
                (N, window["length"][t]*d)
                for t in range(T)
            ]
            assert (len(data_clus) == exp_len)
            for t in range(T):
                assert data_clus[t].shape == exp_shape[t]

        # Not using DTW nor window
        # data_clus is a list of 1 (N, T*d) array
        data_clus = prepare_data(
            data,
            DTW=False, window=None, transformer=None,
            scaler=None,
        )
        assert len(data_clus) == 1
        assert data_clus[0].shape == (N, T*d)

def test_get_clusters():
    # Test with DTW and sliding window
    # data_clus is a list of T (N, w_t, d) arrays

    # Test with DTW but not sliding window
    # data_clus is a list of 1 (N, T, d) array

    # Test without DTW but with sliding window
    # data_clus is a list of T (N, w_t*d) arrays

    # Test without DTW nor sliding window
    # data_clus is a list of 1 (N, T*d) array
    pass

def test_generate_all_clusterings():
    pass

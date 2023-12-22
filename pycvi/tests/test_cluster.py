import numpy as np
from numpy.testing import assert_array_equal
import pytest
from aeon.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List

from ..datasets._mini import mini
from ..cluster import (
    sliding_window, prepare_data, _generate_clustering, generate_all_clusterings,
    generate_uniform
)

def test_generate_uniform():
    """
    Test shapes of the output
    """
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N_zero = 36
        (N, T, d) = data.shape
        l_data0 = generate_uniform(data, N_zero=N_zero)
        assert type(l_data0) == list
        assert len(l_data0) == N_zero
        assert type(l_data0[0]) == np.ndarray
        assert l_data0[0].shape == (N, T, d)

def test_prepare_data():
    """
    Test shapes of the output

    Test with/without DTW/sliding window
    """
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        scaler = StandardScaler().fit(data.reshape(N*T, d))
        l_w = [1, T//2, T]

        # Using DTW and window
        # data_clus is a list of T (N, w_t, d) arrays
        DTW = True
        for w in l_w:
            window = sliding_window(T, w)
            data_clus = prepare_data(
                data,
                DTW=DTW, window=window, transformer=None,
                scaler=scaler,
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
            DTW=DTW, window=None, transformer=None,
            scaler=scaler,
        )
        assert len(data_clus) == 1
        assert data_clus[0].shape == (N, T, d)

        # Not using DTW but using window
        # data_clus is a list of T (N, w_t*d) arrays
        DTW = False
        for w in l_w:
            window = sliding_window(T, w)
            data_clus = prepare_data(
                data,
                DTW=DTW, window=window, transformer=None,
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
            DTW=DTW, window=None, transformer=None,
            scaler=None,
        )
        assert len(data_clus) == 1
        assert data_clus[0].shape == (N, T*d)

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

def test__generate_clustering():
    """
    Test shape and type of output of _generate_clustering

    Test with/without DTW/sliding window
    """
    model_kw = {
        "n_clusters": 2
    }
    k = 2
    model_DTW = TimeSeriesKMeans
    model = KMeans
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        scaler = StandardScaler().fit(data.reshape(N*T, d))
        l_w = [1, T//2, T]

        # Using DTW and window
        # data_clus is a list of T (N, w_t, d) arrays
        DTW = True
        for w in l_w:
            window = sliding_window(T, w)
            data_clus = prepare_data(
                data,
                DTW=DTW, window=window, transformer=None,
                scaler=scaler,
            )
            fit_predict_kw = {
                "X" : data_clus[0]
            }

            clusters = _generate_clustering(
                model_class=model_DTW, model_kw=model_kw,
                fit_predict_kw=fit_predict_kw,
            )

            exp_shape = (N, window["length"][0], d)
            assert (len(clusters) == k)
            # [1: because N differs]
            assert (data_clus[0][clusters[0]].shape[1:] == exp_shape[1:])
            assert (type(clusters[0]) == list)
            assert (type(clusters[0][0]) == int)

        # Using DTW but not window
        # data_clus is a list of 1 (N, T, d) array
        data_clus = prepare_data(
            data,
            DTW=DTW, window=None, transformer=None,
            scaler=scaler,
        )
        fit_predict_kw = {
            "X" : data_clus[0]
        }

        clusters = _generate_clustering(
            model_class=model_DTW, model_kw=model_kw,
            fit_predict_kw=fit_predict_kw,
        )

        exp_shape = (N, T, d)
        assert (len(clusters) == k)
        # [1: because N differs]
        assert (data_clus[0][clusters[0]].shape[1:] == exp_shape[1:])
        assert (type(clusters[0]) == list)
        assert (type(clusters[0][0]) == int)

        # Not using DTW but using window
        # data_clus is a list of T (N, w_t*d) arrays
        DTW = False
        for w in l_w:
            window = sliding_window(T, w)
            data_clus = prepare_data(
                data,
                DTW=DTW, window=window, transformer=None,
                scaler=scaler,
            )
            fit_predict_kw = {
                "X" : data_clus[0]
            }

            clusters = _generate_clustering(
                model_class=model, model_kw=model_kw,
                fit_predict_kw=fit_predict_kw,
            )

            exp_shape = (N, window["length"][0]*d)
            assert (len(clusters) == k)
            # [1: because N differs]
            assert (data_clus[0][clusters[0]].shape[1:] == exp_shape[1:])
            assert (type(clusters[0]) == list)
            assert (type(clusters[0][0]) == int)

        # Not using DTW nor window
        # data_clus is a list of 1 (N, T*d) array
        data_clus = prepare_data(
            data,
            DTW=DTW, window=None, transformer=None,
            scaler=scaler,
        )
        fit_predict_kw = {
            "X" : data_clus[0]
        }

        clusters = _generate_clustering(
            model_class=model, model_kw=model_kw,
            fit_predict_kw=fit_predict_kw,
        )

        exp_shape = (N, T*d)
        assert (len(clusters) == k)
        # [1: because N differs]
        assert (data_clus[0][clusters[0]].shape[1:] == exp_shape[1:])
        assert (type(clusters[0]) == list)
        assert (type(clusters[0][0]) == int)

def test_generate_all_clusterings():
    """
    Test shape and type of output of generate_all_clusterings

    Test with/without DTW/sliding window
    """
    model_DTW = TimeSeriesKMeans
    model = KMeans
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        l_w = [1, T//2, T]

        # Using DTW and window
        # data_clus is a list of T (N, w_t, d) arrays
        for w in l_w:
            clusterings_t_k = generate_all_clusterings(
                data, model_DTW,
                DTW=True, time_window=w, transformer=None,
                scaler=StandardScaler(),
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

            # T_w = 1
            assert (len(clusterings_t_k) == T)
            for k in range(N+1):
                # all clusterings were computed
                assert k in clusterings_t_k[0]
                # k clusters in the clustering k
                exp_len = k
                if k == 0:
                    exp_len = 1
                assert len(clusterings_t_k[0][k]) == exp_len
            # type List[Dict[int, List[List[int]]]]
            assert (type(clusterings_t_k) == list)
            assert (type(clusterings_t_k[0]) == dict)
            assert (type(clusterings_t_k[0][0]) == list)
            assert (type(clusterings_t_k[0][0][0]) == list)
            assert (type(clusterings_t_k[0][0][0][0]) == int)

        # Using DTW but not window
        # data_clus is a list of 1 (N, T, d) array
        clusterings_t_k = generate_all_clusterings(
                data, model_DTW,
                DTW=True, time_window=None, transformer=None,
                scaler=None,
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        # T_w = 1
        for k in range(N+1):
            # all clusterings were computed
            assert k in clusterings_t_k
            # k clusters in the clustering k
            exp_len = k
            if k == 0:
                exp_len = 1
            assert len(clusterings_t_k[k]) == exp_len
        # type Dict[int, List[List[int]]]
        assert (type(clusterings_t_k) == dict)
        assert (type(clusterings_t_k[0]) == list)
        assert (type(clusterings_t_k[0][0]) == list)
        assert (type(clusterings_t_k[0][0][0]) == int)

        # Not using DTW but using window
        # data_clus is a list of T (N, w_t*d) arrays
        for w in l_w:
            clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=False, time_window=w, transformer=None,
                scaler=StandardScaler(),
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

            # T_w = T
            assert (len(clusterings_t_k) == T)
            for k in range(N+1):
                # all clusterings were computed
                assert k in clusterings_t_k[0]
                # k clusters in the clustering k
                exp_len = k
                if k == 0:
                    exp_len = 1
                assert len(clusterings_t_k[0][k]) == exp_len
            # type List[Dict[int, List[List[int]]]]
            assert (type(clusterings_t_k) == list)
            assert (type(clusterings_t_k[0]) == dict)
            assert (type(clusterings_t_k[0][0]) == list)
            assert (type(clusterings_t_k[0][0][0]) == list)
            assert (type(clusterings_t_k[0][0][0][0]) == int)

        # Not using DTW nor window but forcing output to be list
        # data_clus is a list of 1 (N, T*d) array
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=False, time_window=None, transformer=None,
                scaler=None,
                model_kw={}, fit_predict_kw={}, model_class_kw={},
                return_list=True,
            )

        # T_w = 1
        assert (len(clusterings_t_k) == 1)
        for k in range(N+1):
            # all clusterings were computed
            assert k in clusterings_t_k[0]
            # k clusters in the clustering k
            exp_len = k
            if k == 0:
                exp_len = 1
            assert len(clusterings_t_k[0][k]) == exp_len
        # type List[Dict[int, List[List[int]]]]
        assert (type(clusterings_t_k) == list)
        assert (type(clusterings_t_k[0]) == dict)
        assert (type(clusterings_t_k[0][0]) == list)
        assert (type(clusterings_t_k[0][0][0]) == list)
        assert (type(clusterings_t_k[0][0][0][0]) == int)

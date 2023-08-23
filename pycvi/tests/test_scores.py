import numpy as np
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans

from ..datasets import mini
from ..compute_scores import (
    better_score, argbest, best_score, argworst, worst_score,
    compute_score, f_cdist, f_pdist, f_intra, f_inertia, compute_all_scores,
)

def test_comparisons():
    maximize = False
    score1 = [
        -3,    -3,  -3,    0,     2,    -2,    5,    8,    -3,     3,
        None,  None]
    score2 = [
        -4,    -1,   0,    0,    -2,     2,    8,    5,     None,  None,
        2,    -2]
    better_max = [
        False, True, True, False, False, True, True, False, False, False,
        True, True]
    better_min = [
        True, False, False, False, True, False, False, True, False, False,
        True, True]
    for i, (s1, s2) in enumerate(zip(score1, score2)):
        msg = "Better score was wrong, i: {} | s1: {} | s2 {}".format(i, s1, s2)
        out = better_score(s1, s2, maximize)
        assert out == better_max[i], msg
        out = better_score(s1, s2, not maximize)
        assert out == better_min[i], msg

        if better_max[i]:
            out_exp_arg = 0
            out_exp_score = s1
        else:
            out_exp_arg = 1
            out_exp_score = s2
        out = argbest([s1, s2], maximize)
        msg = "argbest was wrong,      i: {} | s1: {} | s2 {}".format(i, s1, s2)
        assert (out == out_exp_arg or s1 == s2), msg


        out = best_score([s1, s2], maximize)
        msg = "best_score was wrong,      i: {} | s1: {} | s2 {}".format(i, s1, s2)
        assert (out == out_exp_score), msg

def test_f_pdist():
    for mulivariate in [True, False]:
        data, time = mini(multivariate=mulivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_pdist(data)
        assert type(dist) == np.ndarray

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_pdist(data)
        assert type(dist) == np.ndarray


def test_f_cdist():
    for mulivariate in [True, False]:
        data, time = mini(multivariate=mulivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_cdist(data[N//2:], data[:N//2])
        assert type(dist) == np.ndarray
        exp_shape = (N-N//2, N//2)
        assert dist.shape == exp_shape

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_cdist(data[N//2:], data[:N//2])
        assert type(dist) == np.ndarray
        exp_shape = (N-N//2, N//2)
        assert dist.shape == exp_shape

def test_f_intra():
    for mulivariate in [True, False]:
        data, time = mini(multivariate=mulivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_intra(data)
        assert type(dist) == float

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_intra(data)
        assert type(dist) == float

def test_f_inertia():
    for mulivariate in [True, False]:
        data, time = mini(multivariate=mulivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_inertia(data)
        assert type(dist) == float

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_inertia(data)
        assert type(dist) == float

def test_compute_subscores():
    pass

def test_compute_score():
    pass
    # Test with DTW and sliding window
    # data_clus is a list of T (N, w_t, d) arrays


    # Test with DTW but not sliding window
    # data_clus is a list of 1 (N, T, d) array

    # Test without DTW but with sliding window
    # data_clus is a list of T (N, w_t*d) arrays

    # Test without DTW nor sliding window
    # data_clus is a list of 1 (N, T*d) array

def test_compute_all_scores():
    pass
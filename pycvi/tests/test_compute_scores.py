import numpy as np
from numpy.testing import assert_array_equal
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..scores import Inertia, GapStatistic

from ..datasets import mini
from ..compute_scores import (
    better_score, argbest, best_score, argworst, worst_score,
    compute_score, f_cdist, f_pdist, f_intra, f_inertia, compute_all_scores,
    compute_subscores
)
from ..cvi import silhouette, CH
from ..cluster import generate_all_clusterings

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
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_pdist(data)
        assert type(dist) == np.ndarray

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_pdist(data)
        assert type(dist) == np.ndarray


def test_f_cdist():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
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
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_intra(data)
        assert type(dist) == float

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_intra(data)
        assert type(dist) == float

def test_f_inertia():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_inertia(data)
        assert type(dist) == float

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_inertia(data)
        assert type(dist) == float

def test_compute_subscores():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        c1 = [i for i in range(N//2)]
        c2 = [i for i in range(N//2, N)]

        # DTW case
        dist = compute_subscores("inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_subscores("max_inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_subscores("list_inertia", data, [c1, c2], "inertia", f_inertia)
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)

        # Non DTW case
        data = data.reshape(N, -1)
        dist = compute_subscores("inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_subscores("max_inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_subscores("list_inertia", data, [c1, c2], "inertia", f_inertia)
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)

def test_compute_score():
    """
    Test types of outputs and try with "str", "Score" and "callable" scores
    """
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        c1 = [i for i in range(N//2)]
        c2 = [i for i in range(N//2, N)]

        # DTW case
        dist = compute_score("inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_score("max_inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_score("list_inertia", data, [c1, c2])
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)
        dist = compute_score(Inertia(), data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_score(silhouette, data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)

        # Non DTW case
        data = data.reshape(N, -1)
        dist = compute_score("inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_score("max_inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_score("list_inertia", data, [c1, c2])
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)
        dist = compute_score(Inertia(), data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = compute_score(silhouette, data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)

def test_compute_all_scores():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape

        # Using DTW but not window
        DTW = True
        model = TimeSeriesKMeans
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=StandardScaler(),
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        scores_t_k = compute_all_scores(
            Inertia(), data, clusterings_t_k,
            transformer=None, scaler=StandardScaler(), DTW=DTW,
            time_window=None,
        )

        # T_w = 1
        assert (len(scores_t_k) == 1)
        for k in range(N+1):
            # all clusterings were computed
            assert k in scores_t_k[0]
        # List[Dict[int, float]]
        assert (type(scores_t_k) == list)
        assert (type(scores_t_k[0]) == dict)
        assert (
            type(scores_t_k[0][0]) == float
            or type(scores_t_k[0][0]) == np.float64
            # returns None when the score was used with an
            # incompatible number of clusters
            or type(scores_t_k[0][0]) == type(None))

        # Not using DTW nor window
        DTW = False
        model = KMeans
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=None,
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        scores_t_k = compute_all_scores(
            GapStatistic(), data, clusterings_t_k,
            transformer=None, scaler=None, DTW=DTW,
            time_window=None
        )

        # T_w = 1
        assert (len(scores_t_k) == 1)
        for k in range(N+1):
            # all clusterings were computed
            assert k in scores_t_k[0]
        # List[Dict[int, float]]
        assert (type(scores_t_k) == list)
        assert (type(scores_t_k[0]) == dict)
        assert (
            type(scores_t_k[0][0]) == float
            or type(scores_t_k[0][0]) == np.float64
            # returns None when the score was used with an
            # incompatible number of clusters
            or type(scores_t_k[0][0]) == type(None))

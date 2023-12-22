import numpy as np
from numpy.testing import assert_array_equal
import pytest
from aeon.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..cvi import Inertia, GapStatistic

from ..datasets._mini import mini
from ..compute_scores import (
    _compute_score, f_intra, f_inertia, compute_all_scores,
    _compute_subscores
)
from ..cvi_func import silhouette, CH
from ..cluster import generate_all_clusterings
from .._utils import _load_data_from_github

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

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
        dist = _compute_subscores("inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_subscores("max_inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_subscores("list_inertia", data, [c1, c2], "inertia", f_inertia)
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)

        # Non DTW case
        data = data.reshape(N, -1)
        dist = _compute_subscores("inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_subscores("max_inertia", data, [c1, c2], "inertia", f_inertia)
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_subscores("list_inertia", data, [c1, c2], "inertia", f_inertia)
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
        dist = _compute_score("inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_score("max_inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_score("list_inertia", data, [c1, c2])
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)
        dist = _compute_score(Inertia(), data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_score(silhouette, data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)

        # Non DTW case
        data = data.reshape(N, -1)
        dist = _compute_score("inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_score("max_inertia", data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_score("list_inertia", data, [c1, c2])
        assert type(dist) == list
        assert (type(dist[0]) == float or type(dist[0]) == np.float64)
        dist = _compute_score(Inertia(), data, [c1, c2])
        assert (type(dist) == float or type(dist) == np.float64)
        dist = _compute_score(silhouette, data, [c1, c2])
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
        # Dict[int, float]
        assert (type(scores_t_k) == dict)
        assert (
            type(scores_t_k[0]) == float
            or type(scores_t_k[0]) == np.float64
            # returns None when the score was used with an
            # incompatible number of clusters
            or type(scores_t_k[0]) == type(None))
        for k in range(N+1):
            # all clusterings were computed
            assert k in scores_t_k

        # Not using DTW nor window but force the output to be list
        DTW = False
        model = KMeans
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=None,
                model_kw={}, fit_predict_kw={}, model_class_kw={},
                return_list=True,
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

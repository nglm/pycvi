import numpy as np
from numpy.testing import assert_array_equal
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..scores import SCORES, Hartigan
from ..datasets import mini
from ..compute_scores import (
    compute_all_scores,
)
from ..cluster import generate_all_clusterings


def test_Scores():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape

        # Using DTW but not window
        model = TimeSeriesKMeans
        DTW = True
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=StandardScaler(),
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        for score in SCORES:
            if score == Hartigan:
                continue
            scores_t_k = compute_all_scores(
                score(), data, clusterings_t_k,
                transformer=None, scaler=StandardScaler(), DTW=DTW,
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

        # Not using DTW nor window
        DTW = False
        model = KMeans
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=None,
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        for score in SCORES:
            if score == Hartigan:
                continue
            scores_t_k = compute_all_scores(
                score(), data, clusterings_t_k,
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

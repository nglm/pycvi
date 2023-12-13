import numpy as np
from numpy.testing import assert_array_equal
import pytest
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from ..scores import SCORES, Score
from ..datasets import mini
from ..compute_scores import (
    compute_all_scores,
)
from ..cluster import generate_all_clusterings
from ..utils import _load_data_from_github

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

def test_Scores():
    # ---------------- Test on toy datasets ----------------------------
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
            for score_type in score.score_types:
                s = score(score_type=score_type)
                scores_t_k = compute_all_scores(
                    s, data, clusterings_t_k,
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
            for score_type in score.score_types:
                s = score(score_type=score_type)
                scores_t_k = compute_all_scores(
                    s, data, clusterings_t_k,
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

    # ---------- Test on clustering benchmark dataset ------------------
    DTW = False
    model = AgglomerativeClustering
    data, meta = _load_data_from_github(PATH + "xclara.arff")
    n_clusters_range = [i for i in range(15)]

    clusterings_t_k = generate_all_clusterings(
            data, model, n_clusters_range=n_clusters_range,
            DTW=DTW, time_window=None, transformer=None,
            scaler=StandardScaler(),
            model_kw={}, fit_predict_kw={}, model_class_kw={}
        )
    for score in SCORES:
        for score_type in score.score_types:
            s = score(score_type=score_type)

            scores_t_k = compute_all_scores(
                s, data, clusterings_t_k,
                transformer=None, scaler=StandardScaler(), DTW=DTW,
                time_window=None
            )

            k = s.select(scores_t_k)

            print(score, k)

def test_is_relevant():
    l_score1 = [-1.,  -1., -1.,  1.,  1.,  1.]
    l_score2 = [-2., -0.5, 0.5, -1., 0.5,  2.]
    # True when s2 is lower
    exp_res_mono_min_imp = [
        s2 < s1 for s1, s2 in zip(l_score1, l_score2)
    ]
    # Typically inertia
    S_monotone = Score(
        maximise=False, improve=True, score_type="monotonous"
    )
    # Typically Silhouette score
    S_abs = Score(
        score_type="absolute"
    )
    k1 = 2
    k2 = 4

    for i, (s1, s2) in enumerate(zip(l_score1, l_score2)):
        assert type(S_abs.is_relevant(s1, k1, s2, k2)) == bool
        assert S_abs.is_relevant(s1, k1, s2, k2)
        assert S_abs.is_relevant(s1, k2, s2, k1)
        assert S_abs.is_relevant(s2, k2, s1, k1)

        assert S_monotone.is_relevant(s1, k1, s2, k2) == exp_res_mono_min_imp[i]
        assert S_monotone.is_relevant(s1, k2, s2, k1) != exp_res_mono_min_imp[i]
        assert S_monotone.is_relevant(s2, k2, s1, k1) == exp_res_mono_min_imp[i]
        assert S_monotone.is_relevant(s2, k1, s1, k2) != exp_res_mono_min_imp[i]

def test_select():
    l_score_mono = [
        {0: 10000, 2: 9000, 4: 5000, 5: 15000, 6: 40000, 7: 9000}
    ]
    l_score_mono_ignore0 = [
        {0: 100, 2: 9000, 4: 5000, 5: 15000, 6: 40000, 7: 9000}
    ]
    l_score_abs  = [ {1: 12000, 2: 5000, 3: 15000, 5: -40000} ]
    # Typically inertia
    S_monotone = Score(
        maximise=False, improve=True, score_type="monotonous"
    )
    # Typically Hartigan
    S_monotone_ignore0 = Score(
        maximise=False, improve=True, score_type="monotonous",
        ignore0=True
    )
    # Typically Silhouette score
    S_abs_max = Score(
        score_type="absolute", maximise=True
    )
    S_abs_min = Score(
        score_type="absolute", maximise=False
    )
    assert S_monotone.select(l_score_mono) == [4]
    assert S_monotone_ignore0.select(l_score_mono_ignore0) == [4]
    assert S_abs_max.select(l_score_abs) == [3]
    assert S_abs_min.select(l_score_abs) == [5]

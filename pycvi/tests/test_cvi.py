import numpy as np
from numpy.testing import assert_array_equal
import pytest
from aeon.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Union

from ._utils import _aux_check_float
from ..cvi import (
    CVIs, CVI, CVIAggregator, Diameter, CalinskiHarabasz, ScoreFunction
)
from ..datasets._mini import mini
from ..compute_scores import (
    compute_all_scores,
)
from ..cluster import generate_all_clusterings
from .._utils import _load_data_from_github
from ..exceptions import SelectionError

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

def _aux_test_selected(cvi, scores_k: Dict[int, float]):
    """
    Check that the select method returns SelectionError when it should.

    Parameters
    ----------
    cvi : CVI
        An instance of a CVI class
    scores_k : Dict[int, float]
        Dict of scores (for a specific t or for non time series data)
    """
    scores_valid = {k: s for k,s in scores_k.items() if s is not None}
    if scores_k == {}:
        try:
            k_selected = cvi.select(scores_k)
        except SelectionError:
            assert True
        else:
            assert False
    elif scores_valid == {} or np.all(np.isclose(
        list(scores_valid.values()), list(scores_valid.values())[0]
    )):
        try:
            k_selected = cvi.select(scores_k)
        except SelectionError:
            assert True
        except:
            assert False
    else:
        k_selected = cvi.select(scores_k)
        assert type(k_selected) == int

def _aux_test_select_aggr(
    aggr,
    scores_i_t_k: Union[List[List[Dict[int, float]]], List[Dict[int, float]]],
):
    # Check in which type case we should be
    if type(scores_i_t_k[0]) == dict:
        should_be_list = False
    else:
        should_be_list = True

    # ---------------- Check the select method -------------------
    # --- Check if SelectionError correctly triggered ---
    try:
        selected_k = aggr.select(scores_i_t_k)
    except SelectionError:
        if (
            (not should_be_list and aggr.votes == {})
            or (should_be_list and {} in aggr.votes)
        ):
            assert True
        else:
            assert False
    # ----------------- Check types -------------------
    else:
        if should_be_list:
            # List[dict]
            assert type(aggr.votes) == list
            assert type(aggr.votes[0]) == dict
            # List[List[int]]
            assert type(aggr.all_selected_k) == list
            assert type(aggr.all_selected_k[0]) == list
            assert type(aggr.all_selected_k[0][0]) == int
            # List[int]
            assert type(selected_k) == list
            assert type(selected_k[0]) == int
        else:
            # dict
            assert type(aggr.votes) == dict
            # List[int]
            assert type(aggr.all_selected_k) == list
            assert type(aggr.all_selected_k[0]) == int
            # int
            assert type(selected_k) == int


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

        for score in CVIs:
            for cvi_type in score.cvi_types:
                s = score(cvi_type=cvi_type)
                scores_t_k = compute_all_scores(
                    s, data, clusterings_t_k,
                    transformer=None, scaler=StandardScaler(), DTW=DTW,
                    time_window=None
                )

                # T_w = 1
                for k in range(N+1):
                    # all clusterings were computed
                    assert k in scores_t_k
                # List[Dict[int, float]]
                assert (type(scores_t_k) == dict)

                # float
                _aux_check_float(scores_t_k[0], or_None=True)

                # int
                _aux_test_selected(s, scores_t_k)

        # Not using DTW nor window but forcing output to be list
        DTW = False
        model = KMeans
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=None,
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        for score in CVIs:
            for cvi_type in score.cvi_types:
                s = score(cvi_type=cvi_type)
                scores_t_k = compute_all_scores(
                    s, data, clusterings_t_k,
                    transformer=None, scaler=None, DTW=DTW,
                    time_window=None, return_list=True,
                )

                # T_w = 1
                assert (len(scores_t_k) == 1)
                for k in range(N+1):
                    # all clusterings were computed
                    assert k in scores_t_k[0]
                # List[Dict[int, float]]
                assert (type(scores_t_k) == list)
                assert (type(scores_t_k[0]) == dict)
                # float
                _aux_check_float(scores_t_k[0][0], or_None=True)

                # List[int]
                for scores_k in scores_t_k:
                    _aux_test_selected(s, scores_k)

                # int
                _aux_test_selected(s, scores_t_k[0])

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
    for score in CVIs:
        for cvi_type in score.cvi_types:
            s = score(cvi_type=cvi_type)

            scores_t_k = compute_all_scores(
                s, data, clusterings_t_k,
                transformer=None, scaler=StandardScaler(), DTW=DTW,
                time_window=None
            )

            # int
            _aux_test_selected(s, scores_t_k)

def test_is_relevant():
    l_score1 = [-1.,  -1., -1.,  1.,  1.,  1.]
    l_score2 = [-2., -0.5, 0.5, -1., 0.5,  2.]
    # True when s2 is lower
    exp_res_mono_min_imp = [
        s2 < s1 for s1, s2 in zip(l_score1, l_score2)
    ]
    # Typically inertia
    S_monotone = CVI(
        maximise=False, improve=True, cvi_type="monotonous"
    )
    # Typically Silhouette score
    S_abs = CVI(
        cvi_type="absolute"
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
    score_mono = {
        0: 10000, 2: 9000, 4: 5000, 5: 15000, 6: 40000, 7: 3000
    }
    l_score_mono_ignore0 = [
        {0: 100, 2: 9000, 4: 5000, 5: 15000, 6: 40000, 7: 9000}
    ]
    score_abs  = {1: 12000, 2: 5000, 3: 15000, 5: -40000}
    # Typically inertia
    S_monotone = CVI(
        maximise=False, improve=True, cvi_type="monotonous"
    )
    # Typically Hartigan
    S_monotone_ignore0 = CVI(
        maximise=False, improve=True, cvi_type="monotonous",
        ignore0=True
    )
    # Typically Silhouette score
    S_abs_max = CVI(
        cvi_type="absolute", maximise=True
    )
    S_abs_min = CVI(
        cvi_type="absolute", maximise=False
    )
    assert S_monotone.select(score_mono) == 4
    assert S_monotone_ignore0.select(l_score_mono_ignore0) == [4]
    assert S_abs_max.select(score_abs, return_list=True) == [3]
    assert S_abs_min.select(score_abs, return_list=True) == [5]

def test_cviaggregator_init():
    list_cvis = [ Diameter, CalinskiHarabasz, ScoreFunction]
    lists_kwargs = [ {"reduction" : "sum"}, {}, {} ]

    aggr = CVIAggregator()
    assert len(aggr.cvis) == len(CVIs)
    assert aggr.n_cvis == len(aggr.cvis)
    assert aggr.n_cvis == len(aggr.cvi_kwargs)
    assert aggr.votes == None
    assert aggr.all_selected_k == None
    assert aggr._is_aggregator == True

    aggr = CVIAggregator(list_cvis)
    assert len(aggr.cvis) == 3
    assert aggr.n_cvis == len(aggr.cvis)
    assert aggr.n_cvis == len(aggr.cvi_kwargs)
    assert aggr.votes == None
    assert aggr.all_selected_k == None
    assert aggr._is_aggregator == True

    aggr = CVIAggregator(list_cvis, lists_kwargs)
    assert len(aggr.cvis) == 3
    assert aggr.n_cvis == len(aggr.cvis)
    assert aggr.n_cvis == len(aggr.cvi_kwargs)
    assert aggr.votes == None
    assert aggr.all_selected_k == None
    assert aggr._is_aggregator == True

    try:
        aggr = CVIAggregator(list_cvis, lists_kwargs[:1])
    except ValueError:
        assert True
    else:
        assert False

def test_cviaggregator():
    # ---------------- Test on toy datasets ----------------------------
    list_cvis = [ Diameter, CalinskiHarabasz, ScoreFunction]
    lists_kwargs = [ {"reduction" : "sum"}, {}, {} ]
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape

        # ========== Using DTW but not window ==========
        model = TimeSeriesKMeans
        DTW = True
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=StandardScaler(),
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        for aggregator in [
            CVIAggregator(), CVIAggregator(list_cvis, lists_kwargs)
        ]:

            scores_i_t_k = compute_all_scores(
                aggregator, data, clusterings_t_k,
                transformer=None, scaler=StandardScaler(), DTW=DTW,
                time_window=None
            )

            # ----------- Testing type of scores_i_t_k -----------------
            # List[dict]
            assert type(scores_i_t_k) == list
            assert (type(scores_i_t_k[0]) == dict)

            for i in range(aggregator.n_cvis):
                for k in range(N+1):
                    # all clustering score were computed
                    assert k in scores_i_t_k[i]
                    _aux_check_float(scores_i_t_k[i][k], or_None=True)

            # ----------- Testing select method ------------------------
            # int
            _aux_test_select_aggr(aggregator, scores_i_t_k)

        # ====== No DTW nor window but forcing list ======
        DTW = False
        model = KMeans
        clusterings_t_k = generate_all_clusterings(
                data, model,
                DTW=DTW, time_window=None, transformer=None,
                scaler=StandardScaler(),
                model_kw={}, fit_predict_kw={}, model_class_kw={}
            )

        for aggregator in [CVIAggregator(), CVIAggregator(list_cvis)]:

            scores_i_t_k = compute_all_scores(
                aggregator, data, clusterings_t_k,
                transformer=None, scaler=StandardScaler(), DTW=DTW,
                time_window=None, return_list=True
            )

            # ----------- Testing type of scores_i_t_k -----------------
            # List[List[dict]]
            assert type(scores_i_t_k) == list
            assert (type(scores_i_t_k[0]) == list)
            assert (type(scores_i_t_k[0][0]) == dict)

            for i in range(aggregator.n_cvis):
                for k in range(N+1):
                    # all clustering score were computed
                    assert k in scores_i_t_k[i][0]
                    _aux_check_float(scores_i_t_k[i][0][k], or_None=True)

            # ----------- Testing select method ------------------------
            # int
            _aux_test_select_aggr(aggregator, scores_i_t_k)

    # ---------- Test on clustering benchmark dataset ------------------
    # ====== No DTW nor window nor forcing list ======
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

    for aggregator in [CVIAggregator(), CVIAggregator(list_cvis)]:

        scores_i_t_k = compute_all_scores(
            aggregator, data, clusterings_t_k,
            transformer=None, scaler=StandardScaler(), DTW=DTW,
            time_window=None
        )

        # ----------- Testing type of scores_i_t_k -----------------
        # List[dict]
        assert type(scores_i_t_k) == list
        assert (type(scores_i_t_k[0]) == dict)

        for i in range(aggregator.n_cvis):
            for k in range(N+1):
                # all clustering score were computed
                assert k in scores_i_t_k[i]
                _aux_check_float(scores_i_t_k[i][k], or_None=True)


        # ----------- Testing select method ------------------------
        # int
        _aux_test_select_aggr(aggregator, scores_i_t_k)
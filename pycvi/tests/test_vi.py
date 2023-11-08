import numpy as np
from numpy.testing import assert_array_equal
import pytest

from ..vi import (
    P_clusters, entropy, contingency_matrix, mutual_information,
    variational_information, align_clusterings,
)
from ..datasets import mini, normal

MARGIN = 1e-6

def clusterings(multivariate=True):
    data, time = normal(multivariate=multivariate)
    C = {}
    N = len(data)
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
    C["C4"] = [[i] for i in range(N)]

    return C, data, time

def test_P_clusters():
    for multivariate in [True, False]:
        C, data, time = clusterings(multivariate=multivariate)
        N = len(data)
        for C1 in C:
            P_ks = P_clusters(C1)
            assert type(P_ks) == list
            assert type(P_ks[0]) == float
            assert len(P_ks) == len(C1)
            assert sum(P_ks) == pytest.approx(1.)
            assert np.all(np.array(P_ks) > 0)

def test_entropy():
    for multivariate in [True, False]:
        Cs, data, time = clusterings(multivariate=multivariate)
        N = len(data)
        for _, C1 in Cs.items():
            H = entropy(C1)
            assert type(H) == float
            assert np.all(H >= 0)

        # entropy when there is only one cluster
        C1 = Cs["C1"]
        H = entropy(C1)
        assert H == 0.

        # entropy when there are only singletons
        C4 = Cs["C4"]
        H = entropy(C4)

def test_contingency_matrix():

    for multivariate in [True, False]:
        Cs, data, time = clusterings(multivariate=multivariate)
        list_keys = list(Cs.keys())
        N = len(data)
        for k1 in list_keys[:-1]:
            C1 = Cs[k1]
            for k2 in list_keys[1:]:
                C2 = Cs[k2]
                # Contingency matrix of two different clusterings
                m = contingency_matrix(C1, C2)
                assert type(m) == np.ndarray
                exp_shape = (len(C1), len(C2))
                assert m.shape == exp_shape
                assert type(m[0, 0]) == np.float64
                assert np.all(m >= 0)
                assert np.sum(m) == pytest.approx(1.)

                # Contingency matrix of two equal clusterings
                m = contingency_matrix(C1, C1)
                assert type(m) == np.ndarray
                exp_shape = (len(C1), len(C1))
                assert m.shape == exp_shape
                assert type(m[0, 0]) == np.float64
                exp_m = np.zeros(exp_shape)
                exp_diag = [len(c)/N for c in C1]
                np.fill_diagonal(exp_m, exp_diag)
                assert_array_equal(exp_m, m)

        # Contingency matrix of non-overlapping clusterings
        C2 = Cs["C2"]
        C2_inv = Cs["C2_inv"]
        exp_shape = (len(C2), len(C2_inv))
        m = contingency_matrix(C2, C2_inv)
        exp_diag = np.zeros(exp_shape).diagonal()
        assert_array_equal(exp_diag, m.diagonal())

def test_mutual_information():
    for multivariate in [True, False]:
        Cs, data, time = clusterings(multivariate=multivariate)
        list_keys = list(Cs.keys())
        N = len(data)
        for k1 in list_keys[:-1]:
            C1 = Cs[k1]
            for k2 in list_keys[1:]:
                # Mutual information of two different clusterings
                C2 = Cs[k2]
                I = mutual_information(C1, C2)
                H1 = entropy(C1)
                H2 = entropy(C2)
                assert type(I) == float
                assert I >= 0 - MARGIN
                assert I <= min(H1, H2) + MARGIN

                # Mutual information of two equal clusterings
                I = mutual_information(C2, C2)
                H = entropy(C2)
                assert I == pytest.approx(H)

                # Mutual information is symmetric
                I = mutual_information(C1, C2)
                I_sym = mutual_information(C2, C1)
                assert I == pytest.approx(I_sym)

def test_variational_information():
    for multivariate in [True, False]:
        Cs, data, time = clusterings(multivariate=multivariate)
        list_keys = list(Cs.keys())
        N = len(data)
        for k1 in list_keys[:-1]:
            C1 = Cs[k1]
            for k2 in list_keys[1:]:
                C2 = Cs[k2]
                # Mutual information of two different clusterings
                vi = variational_information(C1, C2)

                assert type(vi) == float
                # lower bound of vi
                assert vi >= 0 - MARGIN
                # upper bound of vi
                assert vi <= np.log2(N)  + MARGIN

                # vi is symmetric
                vi = variational_information(C1, C2)
                vi_sym = variational_information(C2, C1)
                assert vi == pytest.approx(vi_sym)

                # vi is positive definite
                assert variational_information(C1, C1) == pytest.approx(0)

    # With symmetric clusterings
    C1s = [ Cs["C2_bis"], Cs["C3_bis"], Cs["C3"]]
    C2s = [ Cs["C2_bis_shuffled"], Cs["C3_bis_shuffled"], Cs["C3_shuffled"] ]
    for C1, C2 in zip(C1s, C2s):
        assert variational_information(C1, C1) == pytest.approx(0)

def test_align_clusterings():
    for multivariate in [True, False]:
        Cs, data, time = clusterings(multivariate=multivariate)
        N = len(data)
        C1s = [ Cs["C2_bis"], Cs["C3_bis"] ]
        C2s = [ Cs["C2_bis_shuffled"], Cs["C3_bis_shuffled"] ]
        for C1, C2 in zip(C1s, C2s):

            C1_sorted = sorted(C1, key=len, reverse = True)
            res_c1, res_c2 = align_clusterings(C1_sorted, C2)
            for c1, c2, sorted_c1 in zip(res_c1, res_c2, C1_sorted):
                assert c1 == c2
                assert c1 == sorted_c1

                # type: Tuple[List[List[int]], List[List[int]]]
                assert type(res_c1) == list
                assert type(res_c2) == list
                assert type(res_c1[0]) == list
                assert type(res_c2[0]) == list
                assert type(res_c1[0][0]) == int
                assert type(res_c2[0][0]) == int
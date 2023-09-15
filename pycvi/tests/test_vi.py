import numpy as np
from numpy.testing import assert_array_equal
import pytest

from ..vi import (
    P_clusters, entropy, contingency_matrix, mutual_information,
    variational_information
)
from ..datasets import mini

def clusterings():
    data, time = mini()
    N = len(data)
    C1 = [[i for i in range(N)]]
    C2 = [
        [i for i in range(N//2)],
        [i for i in range(N//2, N)],
    ]
    C2_inv = [
        [i+(N//2) for i in range(N//2)],
        [i-(N//2) for i in range(N//2, N)],
    ]
    C3 = [
        [i for i in range(N//3)],
        [i for i in range(N//3, 2*N//3)],
        [i for i in range(2*N//3, N)],
    ]
    C4 = [[i] for i in range(N)]
    return [C1, C2, C2_inv, C3, C4]

def test_P_clusters():
    Cs = clusterings()
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N = len(data)
        for C1 in Cs:
            P_ks = P_clusters(C1)
            assert type(P_ks) == list
            assert type(P_ks[0]) == float
            assert len(P_ks) == len(C1)
            assert sum(P_ks) == 1.
            assert np.all(np.array(P_ks) > 0)

def test_entropy():
    Cs = clusterings()
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N = len(data)
        for C1 in Cs:
            H = entropy(C1)
            assert type(H) == float
            assert np.all(H >= 0)

    # entropy when there is only one cluster
    C1 = Cs[0]
    H = entropy(C1)
    assert H == 0.

    # entropy when there are only singletons
    N = len(data)
    C4 = Cs[-1]
    H = entropy(C4)
    # log2 is used because the entropy is measured in bits
    assert H == np.log2(N)

def test_contingency_matrix():
    Cs = clusterings()
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N = len(data)
        for C1 in Cs[:-1]:
            for C2 in Cs[1:]:
                # Contingency matrix of two different clusterings
                m = contingency_matrix(C1, C2)
                assert type(m) == np.ndarray
                exp_shape = (len(C1), len(C2))
                assert m.shape == exp_shape
                assert type(m[0, 0]) == np.float64
                assert np.all(m >= 0)
                assert np.sum(m) == 1

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
                C2 = Cs[1]
                C2_inv = Cs[2]
                exp_shape = (len(C2), len(C2_inv))
                m = contingency_matrix(C2, C2_inv)
                exp_diag = np.zeros(exp_shape).diagonal()
                assert_array_equal(exp_diag, m.diagonal())

def test_mutual_information():
    Cs = clusterings()
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N = len(data)
        for C1 in Cs[:-1]:
            for C2 in Cs[1:]:
                # Mutual information of two different clusterings
                I = mutual_information(C1, C2)
                H1 = entropy(C1)
                H2 = entropy(C2)
                assert type(I) == float
                assert I >= 0
                assert I <= min(H1, H2)

                # Mutual information of two equal clusterings
                I = mutual_information(C2, C2)
                H = entropy(C2)
                assert I == H, C2

                # Mutual information is symmetric
                assert mutual_information(C1, C2) == mutual_information(C2, C1)

def test_variational_information():
    Cs = clusterings()
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N = len(data)
        for C1 in Cs[:-1]:
            for C2 in Cs[1:]:
                # Mutual information of two different clusterings
                vi = variational_information(C1, C2)

                assert type(vi) == float
                assert vi >= 0

                # vi is symmetric
                assert (
                    variational_information(C1, C2)
                    == variational_information(C2, C1)
                )

                # vi is positive definite
                assert variational_information(C1, C1) == 0
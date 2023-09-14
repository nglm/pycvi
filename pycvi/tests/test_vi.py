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
                assert type(m[0, 0]) == np.int64
                assert np.all(m >= 0)
                assert np.sum(m) == N

                # Contingency matrix of two equal clusterings
                m = contingency_matrix(C1, C1)
                assert type(m) == np.ndarray
                exp_shape = (len(C1), len(C1))
                assert m.shape == exp_shape
                assert type(m[0, 0]) == np.int64
                exp_m = np.zeros(exp_shape)
                nis = [len(c) for c in C1]
                np.fill_diagonal(exp_m, nis)
                assert_array_equal(exp_m, m)

def test_mutual_information():
    Cs = clusterings()
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        N = len(data)
        for C1 in Cs[:-1]:
            for C2 in Cs[1:]:
                # Mutual information of two different clusterings
                I = mutual_information(C1, C2)
                assert type(I) == float
                assert I > 0

                # Mutual information of two equal clusterings
                I = mutual_information(C2, C2)
                H = entropy(C1)
                assert I == H

    # Mutual information of non-overlapping clusterings
    C2 = Cs[1]
    C2_inv = Cs[2]
    I = mutual_information(C2, C2_inv)
    assert I == 0.


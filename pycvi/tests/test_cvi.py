import numpy as np
import pytest

from ..cvi import (
    _compute_Wk, _clusters_from_uniform
)
from ..utils import load_data_from_github

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

def get_X(
    N:int  = 5,
    d:int  = 2,
    w_t:int  = 3,
) -> np.ndarray:
    return np.random.normal(size=N*d*w_t).reshape((N, w_t, d))

def test__clusters_from_uniform():
    X_DTW = get_X()
    (N, w_t, d) = X_DTW.shape
    X = X_DTW.reshape((N, w_t, d))
    for data in [X, X_DTW]:
        for n_clusters in [1, 2, 3, N]:
            clusters = _clusters_from_uniform(data, n_clusters)
            assert type(clusters) == list
            assert type(clusters[0]) == list
            assert type(clusters[0][0]) == int
            assert int(np.sum([len(c) for c in clusters])) == N


def test__compute_Wk():
    l_T = [1, 3]
    l_d = [1, 2]
    N = 5
    mins = [-1, 0.1]
    maxs = [0.05, 3.]
    l_clusters = [
        [[i for i in range(N)]],
        [[0, 3, 1], [2, 4]],
        [[0, 3, 1, 4], [2]],
    ]
    for d in l_d:
        for T in l_T:
            for DTW in [True, False]:
                for clusters in l_clusters:

                    X = np.random.uniform(
                        low=mins[:d], high= maxs[:d], size=(N, T, d)
                    )
                    if not DTW:
                        X = np.reshape(X, (N, T*d))

                    Wk = _compute_Wk(X, clusters)
                    assert ( type(Wk) == float or type(Wk) == np.float64 )
                    assert Wk >= 0

    data, meta = load_data_from_github(PATH + "xclara.arff")
    N = len(data)
    l_clusters = [
        [[i for i in range(N)]],
        [[i for i in range(N//2)], [i for i in range(N//2, N)]],
        [[i for i in range(N-1)], [N-1]],
    ]
    for clusters in l_clusters:
        Wk = _compute_Wk(data, clusters)
        assert (type(Wk) == float)
        assert Wk >= 0


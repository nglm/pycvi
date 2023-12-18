import numpy as np
from math import isclose
import pytest

from ..cvi_func import (
    _compute_Wk, _clusters_from_uniform, _dist_centroids_to_global,
    _dist_between_centroids, _var
)
from .._utils import _load_data_from_github
from ..datasets._mini import mini, normal, get_clusterings

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

def get_X(
    N:int = 30,
    d:int = 2,
    w_t:int  = 3,
    DTW: bool = False,
) -> np.ndarray:
    if DTW:
        shape = (N, w_t, d)
    else:
        shape = (N, w_t*d)
    return np.random.normal(size=N*d*w_t).reshape(shape)

def test__clusters_from_uniform():
    N = 5
    l_w = [1, 2, 3, 5]
    l_d = [1, 2]
    l_DTW = [True, False]
    for w_t in l_w:
        for d in l_d:
            for DTW in l_DTW:
                X = get_X(N, d, w_t, DTW)
                for n_clusters in [1, 2, 3, N]:
                    clusters = _clusters_from_uniform(X, n_clusters)
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
                    assert (type(Wk) == float)
                    assert Wk >= 0

    data, meta = _load_data_from_github(PATH + "xclara.arff")
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

def test__dist_centroids_to_global():
    N = 30
    C = get_clusterings(N)
    l_w = [1, 2, 3, 5]
    l_d = [1, 2]
    l_DTW = [True, False]
    for w_t in l_w:
        for d in l_d:
            for DTW in l_DTW:
                X = get_X(N, d, w_t, DTW)
                for clusters in C.values():
                    dist = _dist_centroids_to_global(X, clusters)
                    assert type(dist) == list
                    assert type(dist[0]) == float
                    assert len(dist) == len(clusters)

def test__dist_between_centroids():
    N = 30
    C = get_clusterings(N)
    l_w = [1, 2, 3, 5]
    l_d = [1, 2]
    l_DTW = [True, False]
    for w_t in l_w:
        for d in l_d:
            for DTW in l_DTW:
                X = get_X(N, d, w_t, DTW)
                for clusters in C.values():
                    k = len(clusters)

                    dist = _dist_between_centroids(X, clusters, all=False)
                    assert type(dist) == list
                    assert type(dist[0]) == float

                    dist_all = _dist_between_centroids(X, clusters, all=True)
                    assert type(dist_all) == list
                    assert type(dist_all[0]) == list
                    assert type(dist_all[0][0]) == float
                    assert len(dist_all) == k
                    assert len(dist_all[0]) == k
                    assert isclose(2*np.sum(dist), np.sum(dist_all))

def test__var():
    N = 30
    l_w = [1, 2, 3, 5]
    l_d = [1, 2]
    l_DTW = [True, False]
    for w_t in l_w:
        for d in l_d:
            for DTW in l_DTW:
                X = get_X(N, d, w_t, DTW)
                var = _var(X)
                assert type(var) == np.ndarray
                if DTW:
                    assert var.shape == (d, )
                else:
                    assert var.shape == (d*w_t, )
                assert type(var[0]) == np.float64
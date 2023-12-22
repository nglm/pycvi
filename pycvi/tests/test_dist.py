import numpy as np
import pytest

from ..datasets._mini import mini
from ..dist import f_cdist, f_pdist
from .._utils import _load_data_from_github

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

def test_f_pdist():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_pdist(data)
        assert type(dist) == np.ndarray
        assert np.all(dist>=0)

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_pdist(data)
        assert type(dist) == np.ndarray
        assert np.all(dist>=0)
    data, meta = _load_data_from_github(PATH + 'xclara.arff')
    dist = f_pdist(data)
    assert type(dist) == np.ndarray
    assert np.all(dist>=0)


def test_f_cdist():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        # DTW case
        dist = f_cdist(data[N//2:], data[:N//2])
        assert type(dist) == np.ndarray
        assert np.all(dist>=0)
        exp_shape = (N-N//2, N//2)
        assert dist.shape == exp_shape

        # Non DTW case
        data = data.reshape(N, -1)
        dist = f_cdist(data[N//2:], data[:N//2])
        assert np.all(dist>=0)
        assert type(dist) == np.ndarray
        exp_shape = (N-N//2, N//2)
        assert dist.shape == exp_shape

    data, meta = _load_data_from_github(PATH + 'xclara.arff')
    dist = f_cdist(data[:N//2], data[N//2:])
    assert type(dist) == np.ndarray
    assert np.all(dist>=0)

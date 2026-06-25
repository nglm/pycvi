import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering

from ..datasets._mini import mini, mini_2
from ..dist import f_cdist, f_pdist, time_series_metric_with_sklearn
from .._utils import _load_data_from_github

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

def test_f_pdist():
    for multivariate in [True, False]:
        data, time = mini(multivariate=multivariate)
        (N, T, d) = data.shape
        # ts_dist case
        dist = f_pdist(data, dist_kwargs={"window": 0.5})
        assert type(dist) == np.ndarray
        assert np.all(dist>=0)

        # Non ts_dist case
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
        # ts_dist case
        dist = f_cdist(data[N//2:], data[:N//2], dist_kwargs={"window": 0.5})
        assert type(dist) == np.ndarray
        assert np.all(dist>=0)
        exp_shape = (N-N//2, N//2)
        assert dist.shape == exp_shape

        # Non ts_dist case
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

def test_time_series_metric_with_sklearn():
    for multivariate in [True, False]:
        data, time = mini_2(multivariate=multivariate)
        (N, T, d) = data.shape

        # Reshape data to match sklearn requirements
        X = data.reshape(N, T*d)

        # ----- correct k and default dist_kwargs ----
        k = 2

        # Train and predict a AgglomerativeClustering model with a Time-series metric
        model = AgglomerativeClustering(
            n_clusters=k,
            metric=time_series_metric_with_sklearn(X, d=d, T=T),
            linkage="single",
        )

        labels_pred = model.fit_predict(X)

        assert type(labels_pred) == np.ndarray
        assert labels_pred.shape == (N, )

        # ----- incorrect k and dist_kwargs ----
        k = 3

        dist_kwargs = {"method": "msm", "window": 0.5}

        model = AgglomerativeClustering(
            n_clusters=k,
            metric=time_series_metric_with_sklearn(
                X, d=d, T=T, dist_kwargs=dist_kwargs
            ),
            linkage="single",
        )

        labels_pred = model.fit_predict(X)

        assert type(labels_pred) == np.ndarray
        assert labels_pred.shape == (N, )



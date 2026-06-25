import numpy as np

from .._mini import get_clusterings, mini, mini_2, normal


def test_mini():
	data_u, time_u = mini(multivariate=False)
	assert isinstance(data_u, np.ndarray)
	assert isinstance(time_u, np.ndarray)
	assert data_u.shape == (4, 5, 1)
	assert time_u.shape == (5,)

	data_m, time_m = mini(multivariate=True)
	assert isinstance(data_m, np.ndarray)
	assert isinstance(time_m, np.ndarray)
	assert data_m.shape == (4, 5, 2)
	assert time_m.shape == (5,)


def test_mini_2():
	data_u, time_u = mini_2(multivariate=False, time_scale=True)
	assert data_u.shape == (6, 6, 1)
	assert time_u.shape == (6,)
	assert np.array_equal(time_u, np.array([0, 6, 12, 18, 24, 30]))

	data_m, time_m = mini_2(multivariate=True, time_scale=False)
	assert data_m.shape == (6, 6, 2)
	assert np.array_equal(time_m, np.array([0, 1, 2, 3, 4, 5]))



def test_normal():
	k = 3
	nis = [4, 5, 6]
	T = 7

	data, time = normal(
		k=k,
		nis=nis,
		T=T,
		multivariate=True,
		time_scale=True,
	)

	assert isinstance(data, np.ndarray)
	assert isinstance(time, np.ndarray)
	assert data.shape[0] == sum(nis)
	assert data.shape[1] == T
	assert data.shape[2] == 2
	assert time.shape == (T,)


def test_get_clusterings():
	N = 12
	clusterings = get_clusterings(N)

	expected_keys = {
		"C1", "C2", "C2_bis", "C2_inv", "C2_shuffled",
		"C2_bis_shuffled", "C3", "C3_bis", "C3_shuffled",
		"C3_bis_shuffled", "CN"
	}
	assert expected_keys.issubset(set(clusterings.keys()))

	# C1 contains all points in one cluster.
	assert len(clusterings["C1"]) == 1
	assert len(clusterings["C1"][0]) == N

	# CN contains singleton clusters.
	assert len(clusterings["CN"]) == N
	assert all(len(c) == 1 for c in clusterings["CN"])

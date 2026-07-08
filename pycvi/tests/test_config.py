import numpy as np
import pytest
from numpy.random import Generator, RandomState
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ..config import (
	_get_model_parameters,
	set_random_state,
	default_ts_average_kwargs,
	default_ts_distance_kwargs,
	set_data_shape,
)
from ..exceptions import ShapeError


@pytest.mark.parametrize(
	"data, expected_shape",
	[
		(np.array([1.0, 2.0]), (2, 1, 1)),
		(np.array([[1.0, 2.0], [3.0, 4.0]]), (2, 1, 2)),
		(np.arange(12.0).reshape(2, 3, 2), (2, 3, 2)),
	],
)
def test_set_data_shape_valid_inputs(data, expected_shape):
	reshaped = set_data_shape(data)

	assert isinstance(reshaped, np.ndarray)
	assert reshaped.shape == expected_shape
	assert not np.shares_memory(reshaped, data)


def test_set_data_shape_rejects_single_sample():
	with pytest.raises(ShapeError, match="At least 2 samples"):
		set_data_shape(np.array([1.0]))


def test_set_data_shape_rejects_invalid_number_of_dimensions():
	with pytest.raises(ShapeError, match="Invalid shape"):
		set_data_shape(np.zeros((2, 1, 1, 1)))


def test_set_random_state():

    # Check that it returns the same object when input and rtype matches
	generator = np.random.default_rng(12)
	random_state = np.random.RandomState(12)

	assert set_random_state(generator, Generator) is generator
	assert set_random_state(random_state, RandomState) is random_state
	assert set_random_state(12, int) == 12

    # Check that it returns the correct type when input and rtype differ
	generator_from_int = set_random_state(12, Generator)
	random_state_from_int = set_random_state(12, RandomState)
	int_from_generator = set_random_state(np.random.default_rng(12), int)
	generator_from_random_state = set_random_state(
		np.random.RandomState(12), Generator
	)
	random_state_from_generator = set_random_state(
        np.random.default_rng(12), RandomState
    )

	assert isinstance(generator_from_int, Generator)
	assert isinstance(random_state_from_int, RandomState)
	assert isinstance(int_from_generator, int)
	assert isinstance(generator_from_random_state, Generator)
	assert isinstance(random_state_from_generator, RandomState)

    # Check that errors are raised for invalid types
	with pytest.raises(ValueError, match="Invalid RandomGenerator type"):
		set_random_state("seed", Generator)

	with pytest.raises(ValueError, match="Invalid RandomGenerator return type"):
		set_random_state(12, float)


def test__get_model_parameters_for_kmeans_applies_defaults_and_overrides():
	model_kw = {
		"max_iter": 200,
		"n_init": 5,
		"algorithm": "lloyd",
	}
	fit_predict_kw = {"sample_weight": np.array([1.0, 2.0])}
	model_class_kw = {"X_arg_name": "data"}

	m_kw, ft_kw, mc_kw = _get_model_parameters(
		KMeans,
		model_kw=model_kw,
		fit_predict_kw=fit_predict_kw,
		model_class_kw=model_class_kw,
	)

	assert m_kw == {
		"max_iter": 200,
		"n_init": 5,
		"tol": 1e-3,
		"algorithm": "lloyd",
	}
	assert ft_kw == fit_predict_kw
	assert mc_kw == {
		"k_arg_name": "n_clusters",
		"X_arg_name": "data",
	}


def test__get_model_parameters_for_gaussian_mixture_uses_component_key():
	m_kw, ft_kw, mc_kw = _get_model_parameters(
		GaussianMixture,
		model_kw={"covariance_type": "diag"},
		fit_predict_kw={"y": None},
		model_class_kw={"X_arg_name": "values"},
	)

	assert m_kw == {"covariance_type": "diag"}
	assert ft_kw == {"y": None}
	assert mc_kw == {
		"k_arg_name": "n_components",
		"X_arg_name": "values",
	}


def test_default_ts_average_kwargs_adds_defaults_without_mutating_input():
	user_kwargs = {"method": "subgradient", "window": 0.25}

	final_kwargs = default_ts_average_kwargs(user_kwargs)

	assert final_kwargs == {
		"distance": "msm",
		"init_barycenter": "medoids",
		"method": "subgradient",
		"random_state": 221,
		"window": 0.25,
	}
	assert user_kwargs == {"method": "subgradient", "window": 0.25}


def test_default_ts_distance_kwargs_defaults_to_msm_method():
	user_kwargs = {"window": 0.5}

	final_kwargs = default_ts_distance_kwargs(user_kwargs)

	assert final_kwargs == {"method": "msm", "window": 0.5}
	assert user_kwargs == {"window": 0.5}


@pytest.mark.parametrize(
	"user_kwargs",
	[
		{"CALLABLE": lambda x, y: 0.0, "window": 0.5},
		{"method": "dtw", "window": 0.5},
	],
)
def test_default_ts_distance_kwargs_skips_default_method_when_overridden(user_kwargs):
	final_kwargs = default_ts_distance_kwargs(user_kwargs)

	assert final_kwargs == user_kwargs

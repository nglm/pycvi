# PyCVI 0.1.6 release notes

2026/02/XX

## Python versions

This version supports Python versions 3.9 to 3.12.

## New Features

- Can now use all the parameters of the distance function used to compute distances between datapoints in the CVI functions, by using the ``dist_kwargs`` parameter in the functional API and the ``cvi_kwargs`` parameter in the object-oriented API.
  - If the dataset ``X`` is time-series data and if DTW is used, then the ``dist_kwargs`` can include parameters such as ``window`` or ``itakura_max_slope``. See [aeon.distances.dtw_pairwise_distance](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distance) for more information.
  - Otherwise, the distance function used is based on [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) and accepts the same parameters as this function.
- Can now compute cluster centers using different distance argument for DTW.
- Added function ``pcvi.cluster.compute_centers`` to compute all the cluster centers at once, with distance function parameters.
- Can now use the ``keepdims`` parameter when computing cluster centers, to keep the ``N`` dimension when computing the centers if desired.

## Changes

- Now uses UV to manage the dependencies and build the package.

## Fixes

- XXX

## Documentation

- Added installation instructions using UV.
- Added example and documentation about the ``__call__`` method of the CVI classes.
- More examples about how to use the CVIs and their parameters, notably using ``dist_kwargs`` in the functional API and ``cvi_kwargs`` in the object-oriented API.

## Contributors

- Natacha Galmiche (@nglm)
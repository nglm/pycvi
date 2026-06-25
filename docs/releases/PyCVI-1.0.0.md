# PyCVI 1.0.0 release notes

2026/06/23

## Python versions

This version supports Python versions 3.11 to 3.14.

## New Features

- `pycvi.dist.f_pdist`, `pycvi.dist.f_cdist` now accept a wide range of distance functions and parameters, both for static and time-series data, leveraging `scipy` and `aeon` packages.
- `pycvi.dist.f_pdist`, `pycvi.dist.f_cdist` now accept custom callable.
- `pycvi.cluster.compute_center` now allow a wider range of parameters for the elastique averaging computation.
- `pycvi.dist.time_series_metric_with_sklearn` allows the user to use sklearn models (and sklearn-like models) with distance measures designed for time-series.

## Changes

- Examples are no longer featuring compatibility with sklearn-extra, because sklearn-extra is not maintained anymore (and becoming incompatible with everything else).
- The `DTW` parameter, which determines whether a time-series distance such as DTW should be used is now called `ts_dist`
- By default, `MSM` and `MBA` are now used instead of `DTW` and `DBA`.
- In `pycvi.cluster.compute_center` and `pycvi.cluster.compute_centers`, the `dist_kwargs` parameter is now called `avg_kwargs`
- In the functional API, all CVI functions now take an additional optional parameter `avg_kwargs` that will be sent to `pycvi.cluster.compute_center` and `pycvi.cluster.compute_centers` (and then `aeon.clustering.averaging.elastic_barycenter_average`).
- In the OOP API, all CVI classes can have in their `cvi_kwargs` a `avg_kwargs` key containing the dictionary of kwargs for the average function.

## Contributors

- Natacha Galmiche (@nglm)
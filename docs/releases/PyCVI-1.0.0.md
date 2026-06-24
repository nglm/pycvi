# PyCVI 1.0.0 release notes

2026/06/23

## Python versions

This version supports Python versions 3.11 to 3.14.

## New Features

- `pycvi.dist.f_pdist`, `pycvi.dist.f_cdist` now accept a wide range of distance functions and parameters, both for static and time-series data, leveraging `scipy` and `aeon` packages.
- `pycvi.dist.f_pdist`, `pycvi.dist.f_cdist` now accept custom callable.
- `pycvi.cluster.compute_center` now allow a wider range of parameters for the elastique averaging computation.

## Changes

- Examples are not featuring compatibility with sklearn-extra, because sklearn-extra is not maintained anymore (and becoming incompatible with everything else).
- The `DTW` parameter, which determines whether a time-series distance such as DTW should be used is now called `ts_dist`
- By default, `MSM` and `MBA` are now used instead of `DTW` and `DBA`.

## Contributors

- Natacha Galmiche (@nglm)
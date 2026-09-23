# PyCVI 1.2.0 release notes

2026/XX/XX

## Python versions

This version supports Python versions 3.11 to 3.14.

## New Features

- Adding compatibility with clustering methods that don't take the number of cluster k as main parameter such as OPTICS, DBSCAN, HDBSCAN, AffinityPropagation, etc.
- Adding compatibility with sklearn-like clustering methods that accept a metric parameter and don't necessarily take k as their main parameter with time series data.
- Adding full compatibility with `kmedoids`.
- Full control over random_state and the random state can be an int, a RandomState or a Generator.
- CVIs, `compute_all_scores`, `generate_all_clusterings` can now be used with clustering methods that don't necessarily take the number of clusters k as their main parameter.
- CVIs that are monotonous now raise a ValueError if they are used with clusterings where the number of clusters k is not the main parameter.

## Changes

- `time_series_metric_with_sklearn` now supports a wider range of sklearn-like models and doesn't take the data `X` as a parameter.

## Contributors

- Natacha Galmiche (@nglm)
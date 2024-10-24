# PyCVI 0.1.5 release notes

2024/10/23

## Python versions

This version supports Python versions 3.8 to 3.11.

## New Features

- `pycvi.compute_scores.compute_all_scores`, CVIs and their corresponding functions can now use a specific numpy [Random Generator](https://numpy.org/doc/stable/reference/random/generator.html) for reproducibility purpose.

## Changes

- `pycvi.cvi.select` now raises a `SelectionError` if no clustering could be selected (was `None` previously)

## Fixes

- Fix Davies-Bouldin in order to ignore comparing the distance between a centroid to itself when computing the index.

## Documentation

- Added "how to cite" section
- Fixed truncated examples

## Contributors

- Natacha Galmiche (@nglm)
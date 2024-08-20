# PyCVI 0.1.3 release notes

2024/08/20

## Python versions

This version supports Python versions 3.8 to 3.11.

## New Features

- The class [`pycvi.cvi.CVIAggregator`](https://pycvi.readthedocs.io/en/latest/pycvi.cvi.html#pycvi.cvi.CVIAggregator) was added in order to be able to easily combine several CVIs to select the best clustering. An example featuring `pycvi.cvi.CVIAggregator` has been added to the documentation: [CVIAggregator: Combining CVIs](https://pycvi.readthedocs.io/en/latest/examples/cvi_aggregator.html)
- [`pycvi.cvi.compute_all_scores`](https://pycvi.readthedocs.io/en/latest/pycvi.compute_scores.compute_all_scores.html) can now take either a [CVI](https://pycvi.readthedocs.io/en/latest/pycvi.cvi.html#pycvi.cvi.CVI) or a [CVIAggregator](https://pycvi.readthedocs.io/en/latest/pycvi.cvi.html#pycvi.cvi.CVIAggregator) as an argument.

## Changes

- `pycvi.cvi.select` method now raises a `SelectionError` when all cvi values were equal instead of choosing the clustering with the smallest number of clusters.

## Fixes

- Some package compatibility issues fixed
- Minor fixes in the documentation

## Contributors

- Natacha Galmiche (@nglm)
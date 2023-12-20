# PyCVI

[![Python package](https://github.com/nglm/pycvi/actions/workflows/python-package.yml/badge.svg)](https://github.com/nglm/pycvi/actions/workflows/python-package.yml)

Python package specialized in internal Clustering Validity Indices (CVI), in order to select the best clustering among a set of pre-computed clusterings when no external information such as the labels of the datapoints is available.

## Features

- 11 CVIs implemented: Hartigan, Calinski-Harabasz, GapStatistic, Silhouette, ScoreFunction, Maulik-Bandyopadhyay, SD, SDbw, Dunn, Xie-Beni and Davies-Bouldin.
- Compute CVI values and select the best clustering based on the results.
- Compatible with time-series, Dynamic Time Warping (DTW) and Dynamic time warping Barycentric Average (DBA)[^1].
- Compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), [scikit-learn extra](https://scikit-learn-extra.readthedocs.io/en/stable/) and [tslearn](https://tslearn.readthedocs.io).
- Can compute the clusterings beforehand if provided with a sklearn-like clustering class.
- Enable users to define custom CVIs.

## Install

- With poetry:

```bash
poetry add git+https://github.com/nglm/pycvi.git
```

- With pip:

```bash
pip install git+https://github.com/nglm/pycvi.git
```

## Contribute

- Issue Tracker: [github.com/nglm/pycvi/issues](https://github.com/nglm/pycvi/issues).
- Source Code: [github.com/nglm/pycvi](github.com/nglm/pycvi).

## Support

If you are having issues, [please let me know](https://www.uib.no/en/persons/Natacha.Madeleine.Georgette.Galmiche).

## License

The project is licensed under the MIT license.

[^1]:  F. Petitjean, A. Ketterlin, and P. Gan carski, “A global averaging method for dynamic time warping, with applications to clustering,” *Pattern Recognition*, vol. 44, pp. 678–693, Mar. 2011.
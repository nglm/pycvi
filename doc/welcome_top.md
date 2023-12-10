# PyCVI

PyCVI is a Python package specialized in internal Clustering Validity Indices (CVI). CVIs are used to select the best clustering among a set of pre-computed clusterings when no external information such as the labels of the datapoints is available.

## Features

- 11 CVIs implemented: Hartigan, Calinski-Harabasz, GapStatistic, Silhouette, ScoreFunction, Maulik-Bandyopadhyay, SD, SDbw, Dunn, Xie-Beni and Davies-Bouldin.
- Compute CVI values and select the best clustering based on the results.
- Compatible with time-series, Dynamic Time Warping (DTW) and Dynamic time warping Barycentric Average (DBA).
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


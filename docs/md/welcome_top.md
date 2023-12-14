# Welcome to PyCVI's documentation!

PyCVI is a Python package specialized in internal Clustering Validity Indices (CVI). Internal CVIs are used to select the best clustering among a set of pre-computed clusterings when no external information is available such as the labels of the datapoints.

All CVIs rely on the definition of a distance between datapoints and most of them on the notion of cluster center.

For non-time-series data, the distance used is usually the euclidean distance and the cluster center is defined as the usual average. Libraries such as `scipy`, `numpy`, `sklearn`, etc. offer a large selection of distance measures that are compatible with all their functions.

For time-series data however, the common distance used is Dynamic Time Warping (DTW) and the barycenter of a group of time series is then not defined as the usual mean, but as the DTW Barycentric Average (DBA). Unfortunately, DTW and DBA are not compatible with the libraries mentioned above.

In addition, although being fundamental to clustering tasks and being an active research topic, very few internal CVIs are implemented in standard python libraries (only 3 in `sklearn`, more were available in R but few were maintained and kept in CRAN). This is despite the well-known limitations of all existing CVIs and the need to use the right one(s) according to the specific dataset at hand.

PyCVI then tries to fill that gap by implementing 11 state-of-the-art internal CVIs that are compatible with DTW and DBA (and obviously non time-series data). PyCVI is entirely compatible with `sklearn`, `tslearn` and `skearn_extra`, in order to be easily integrated into any clustering pipeline in python.


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


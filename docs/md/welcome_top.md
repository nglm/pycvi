# Welcome to PyCVI's documentation!

[![PyPI version](https://badge.fury.io/py/pycvi-lib.svg)](https://badge.fury.io/py/pycvi-lib)
[![Python package](https://github.com/nglm/pycvi/actions/workflows/python-package.yml/badge.svg)](https://github.com/nglm/pycvi/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/pycvi/badge/?version=latest)](https://pycvi.readthedocs.io/en/latest/?badge=latest)
[![status](https://joss.theoj.org/papers/fb63888e0a78da2866b03247ce18909d/status.svg)](https://joss.theoj.org/papers/fb63888e0a78da2866b03247ce18909d)

PyCVI is a Python package specialized in internal Clustering Validity Indices (CVI). Internal CVIs are used to select the best clustering among a set of pre-computed clusterings when no external information is available such as the labels of the datapoints.

Although being fundamental to clustering tasks and being an active research topic, very few internal CVIs are implemented in standard python libraries (only 3 in [scikit-learn](https://scikit-learn.org/stable/index.html), more were available in R but few were maintained and kept in CRAN). This is despite the well-known limitations of all existing CVIs and the need to use the right one(s) according to the specific dataset at hand.

In addition, all CVIs rely on the definition of a distance between datapoints and most of them on the notion of cluster center.

For non-time-series data, the distance used is usually the euclidean distance and the cluster center is defined as the usual average. Libraries such as [scipy](https://docs.scipy.org/doc/scipy/index.html), [numpy](https://numpy.org/doc/stable/), [scikit-learn](https://scikit-learn.org/stable/index.html), etc. offer a large selection of distance measures that are compatible with all their functions.

For time-series data however, the common distance used is Dynamic Time Warping (DTW) and the barycenter of a group of time series is then not defined as the usual mean, but as the DTW Barycentric Average (DBA). Unfortunately, DTW and DBA are not compatible with the libraries mentioned above, which among other reasons, made additional machine learning libraries specialized in time series data such as [aeon](https://www.aeon-toolkit.org/en/latest/index.html), [sktime](https://www.sktime.net/en/stable/index.html) and [tslearn](https://tslearn.readthedocs.io/en/stable/) necessary.

PyCVI then tries to fill that gap by implementing 12 state-of-the-art internal CVIs and by making them compatible with DTW and DBA (and obviously non time-series data). PyCVI is entirely compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), [scikit-learn extra](https://scikit-learn-extra.readthedocs.io/en/stable/), [aeon](https://www.aeon-toolkit.org/en/latest/index.html) and [sktime](https://www.sktime.net/en/stable/index.html), in order to be easily integrated into any clustering pipeline in python.

To compute DTW and DBA, PyCVI relies on the [aeon](https://www.aeon-toolkit.org/en/latest/index.html) library.

## Main features

- 12 internal CVIs implemented: Hartigan, Calinski-Harabasz, GapStatistic, Silhouette, ScoreFunction, Maulik-Bandyopadhyay, SD, SDbw, Dunn, Xie-Beni, XB* and Davies-Bouldin.
- Compute CVI values and select the best clustering based on the results.
- Compatible with time-series, Dynamic Time Warping (DTW) and Dynamic time warping Barycentric Average (DBA).
- Compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), [scikit-learn extra](https://scikit-learn-extra.readthedocs.io/en/stable/), [aeon](https://www.aeon-toolkit.org/en/latest/index.html) and [sktime](https://www.sktime.net/en/stable/index.html), for an easy integration into any clustering pipeline in python.
- Can compute the clusterings beforehand if provided with a sklearn-like clustering class.
- Enable users to define custom CVIs.
- Multiple CVIs can easily be combined to select the best clustering based on a majority vote.
- Variation of Information implemented (distances between clustering).

## Install

### With poetry

```bash
# From PyPI
poetry add pycvi-lib
# Alternatively, from github directly
poetry add git+https://github.com/nglm/pycvi.git
```

### With pip

```bash
# From PyPI
pip install pycvi-lib
# Alternatively, from github directly
pip install git+https://github.com/nglm/pycvi.git
```

### With anaconda

```bash
# activate your environment (replace myEnv with your environment name)
conda activate myEnv
# install pip first in your environment
conda install pip
# install pycvi on your anaconda environment with pip
pip install pycvi-lib
```

### Extra dependencies

In order to run the example scripts, extra dependencies are necessary. The install command is then:

```bash
# For poetry
poetry add pycvi-lib[examples]
# For pip and anaconda
pip install pycvi-lib[examples]
```

Alternatively, you can manually install in your environment the packages that are necessary to run the example scripts (`matplotlib` and/or `scikit-learn-extra` depending on the example).
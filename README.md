# PyCVI

[![PyPI version](https://badge.fury.io/py/pycvi-lib.svg)](https://badge.fury.io/py/pycvi-lib)
[![Python package](https://github.com/nglm/pycvi/actions/workflows/python-package.yml/badge.svg)](https://github.com/nglm/pycvi/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/pycvi/badge/?version=latest)](https://pycvi.readthedocs.io/en/latest/?badge=latest)
[![status](https://joss.theoj.org/papers/fb63888e0a78da2866b03247ce18909d/status.svg)](https://joss.theoj.org/papers/fb63888e0a78da2866b03247ce18909d)

PyCVI is a Python package specialized in internal Clustering Validity Indices (CVI). Internal CVIs are used to select the best clustering among a set of pre-computed clusterings when no external information is available such as the labels of the datapoints.

In addition, all CVIs rely on the definition of a distance between datapoints and most of them on the notion of cluster center.

For non-time-series data, the distance used is usually the euclidean distance and the cluster center is defined as the usual average. Libraries such as [scipy](https://docs.scipy.org/doc/scipy/index.html), [numpy](https://numpy.org/doc/stable/), [scikit-learn](https://scikit-learn.org/stable/index.html), etc. offer a large selection of distance measures that are compatible with all their functions.

For time-series data however, the common distance used is Dynamic Time Warping (DTW) [^DTW] and the barycenter of a group of time series is then not defined as the usual mean, but as the DTW Barycentric Average (DBA)[^DBA]. Unfortunately, DTW and DBA are not compatible with the libraries mentioned above, which among other reasons, made additional machine learning libraries specialized in time series data such as [aeon](https://www.aeon-toolkit.org/en/latest/index.html), [sktime](https://www.sktime.net/en/stable/index.html) and [tslearn](https://tslearn.readthedocs.io/en/stable/) necessary.

PyCVI then implements 12 state-of-the-art internal CVIs and extended them to make them compatible with DTW and DBA when using time-series data. To compute DTW and DBA, PyCVI relies on the [aeon](https://www.aeon-toolkit.org/en/latest/index.html) library.

## Documentation

The full documentation is available at [pycvi.readthedocs.io](https://pycvi.readthedocs.io/en/latest/).

## Features

- 12 internal CVIs implemented: Hartigan[^Hart], Calinski-Harabasz[^CH], GapStatistic[^Gap], Silhouette[^Sil], ScoreFunction[^SF], Maulik-Bandyopadhyay[^MB], SD[^SD], SDbw[^SDbw], Dunn[^D], Xie-Beni[^XB], XB*[^XB*] and Davies-Bouldin[^DB].
- Compute CVI values and select the best clustering based on the results.
- Compatible with time-series, Dynamic Time Warping (DTW) and Dynamic time warping Barycentric Average (DBA).
Compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), [scikit-learn extra](https://scikit-learn-extra.readthedocs.io/en/stable/), [aeon](https://www.aeon-toolkit.org/en/latest/index.html) and [sktime](https://www.sktime.net/en/stable/index.html), for an easy integration into any clustering pipeline in python.
- Can compute the clusterings beforehand if provided with a sklearn-like clustering class.
- Enable users to define custom CVIs.
- Multiple CVIs can easily be combined to select the best clustering based on a majority vote.
- Variation of Information[^VI] implemented (distances between clustering).

## Install

### With poetry:

```bash
# From PyPI
poetry add pycvi-lib
# Alternatively, from github directly
poetry add git+https://github.com/nglm/pycvi.git
```

### With pip:

```bash
# From PyPI
pip install pycvi-lib
# Alternatively, from github directly
pip install git+https://github.com/nglm/pycvi.git
```

### With anaconda:

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

If you wish to run the example scripts on your own computer, please follow the instructions detailed in the documentation first: [Running example scripts on your computer](https://pycvi.readthedocs.io/en/latest/examples/examples_instructions.html).

## Contribute

- Issue Tracker: [github.com/nglm/pycvi/issues](https://github.com/nglm/pycvi/issues).
- Source Code: [github.com/nglm/pycvi](github.com/nglm/pycvi).

## Support

If you are having issues, [please let me know](https://www.uib.no/en/persons/Natacha.Madeleine.Georgette.Galmiche) or [create an issue](https://github.com/nglm/pycvi/issues).

## License

The project is licensed under the MIT license.

## How to cite PyCVI

If you are using PyCVI in your work, please cite us by using one of the following entries referring to the JOSS paper "*PyCVI: A Python package for internal Cluster Validity Indices, compatible with time-series data*" by N. Galmiche:

**BibTeX**

```tex
@article{Galmiche2024,
    author = {Natacha Galmiche},
    title = {PyCVI: A Python package for internal Cluster Validity Indices, compatible with time-series data},
    doi = {10.21105/joss.06841},
    url = {https://doi.org/10.21105/joss.06841},
    year = {2024},
    publisher = {The Open Journal},
    volume = {9},
    number = {102},
    pages = {6841},
    journal = {Journal of Open Source Software}
}
```

**Plain text**

```text
Galmiche, N., (2024). PyCVI: A Python package for internal Cluster Validity Indices, compatible with time-series data. Journal of Open Source Software, 9(102), 6841, https://doi.org/10.21105/joss.06841
```

[^DTW]: Donald J. Berndt and James Clifford. Using dynamic time warping to find patterns in time series. In Proceedings of the 3rd International Conference on Knowledge Discovery and Data Mining, AAAIWS’94, page 359–370. AAAI Press, 1994
[^DBA]:  F. Petitjean, A. Ketterlin, and P. Gan carski, “A global averaging method for dynamic time warping, with applications to clustering,” *Pattern Recognition*, vol. 44, pp. 678–693, Mar. 2011.
[^Hart]: D. J. Strauss and J. A. Hartigan, “Clustering algorithms,”
*Biometrics*, vol. 31, p. 793, sep 1975.
[^CH]: T. Calinski and J. Harabasz, “A dendrite method for cluster analysis,” *Communications in Statistics - Theory and Methods*, vol. 3, no. 1, pp. 1–27, 1974.
[^Gap]: R. Tibshirani, G. Walther, and T. Hastie, “Estimating the number of clusters in a data set via the gap statistic,” *Journal of the Royal Statistical Society Series B: Statistical* Methodology, vol. 63, pp. 411–423, July 2001.
[^Sil]: P. J. Rousseeuw, “Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,” *Journal of computational and applied mathematics*, vol. 20, pp. 53–65, 1987.
[^D]: J. C. Dunn, “Well-separated clusters and optimal fuzzy partitions,” *Journal of Cybernetics*, vol. 4, pp. 95–104, Jan. 1974.
[^DB]: D. L. Davies and D. W. Bouldin, “A cluster separation measure,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-1, pp. 224–227, Apr. 1979.
[^SD]: M. Halkidi, M. Vazirgiannis, and Y. Batistakis, “Quality scheme assessment in the clustering process,” in *Principles of Data Mining and Knowledge Discovery*, pp. 265–276, Springer Berlin Heidelberg, 2000
[^SDbw]: M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: finding the optimal partitioning of a data set,” in *Proceedings 2001 IEEE International Conference on Data Mining*, pp. 187–194, IEEE Comput. Soc, 2001.
[^XB]: X. Xie and G. Beni, “A validity measure for fuzzy clustering,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 13, no. 8, pp. 841–847, 1991.
[^XB*]: M. Kim and R. Ramakrishna, “New indices for cluster validity assessment,” *Pattern Recognition Letters*, vol. 26, pp. 2353–2363, Nov. 2005.
[^SF]: S. Saitta, B. Raphael, and I. F. C. Smith, “A bounded index for cluster validity,” in *Machine Learning and Data Mining in Pattern Recognition*, pp. 174–187, Springer Berlin Heidelberg, 2007.
[^MB]: U. Maulik and S. Bandyopadhyay, “Performance evaluation of some clustering algorithms and validity indices,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 24, pp. 1650–1654, Dec. 2002.
[^VI]: M. Meil ̆a, Comparing Clusterings by the Variation of Information, p. 173–187. Springer Berlin Heidelberg, 2003.
# PyCVI 0.1.0 release notes

2023/12/22

First release of PyCVI.

## Python versions

This version supports Python versions 3.8 to 3.11.

## Features

- 12 internal CVIs implemented: Hartigan, Calinski-Harabasz, GapStatistic, Silhouette, ScoreFunction, Maulik-Bandyopadhyay, SD, SDbw, Dunn, Xie-Beni, XB* and Davies-Bouldin.
- Compute CVI values and select the best clustering based on the results.
- Compatible with time-series, Dynamic Time Warping (DTW) and Dynamic time warping Barycentric Average (DBA).
- Compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), [scikit-learn extra](https://scikit-learn-extra.readthedocs.io/en/stable/), [aeon](https://www.aeon-toolkit.org/en/latest/index.html) and [sktime](https://www.sktime.net/en/stable/index.html), for an easy integration into any clustering pipeline in python.
- Can compute the clusterings beforehand if provided with a sklearn-like clustering class.
- Enable users to define custom CVIs.
- Variation of Information implemented (distances between clustering).
- counterparts of [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) and [cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) in scipy, for time series and non time-series data.
- DTW and DBA based on [aeon](https://www.aeon-toolkit.org/en/latest/index.html).

## Contributors

- Natacha Galmiche (@nglm)
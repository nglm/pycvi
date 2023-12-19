---
title: 'PyCVI: A Python package for internal Cluster Validity Indices, compatible with time-series data'
tags:
  - Python
  - scikit-learn
  - sklearn
  - machine learning
  - cluster validity index
  - CVI
  - clustering
  - time series
  - DTW
  - DBA
authors:
  - name: Natacha Galmiche
    orcid: 0000-0002-8473-1673
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: University of Bergen, Norway
   index: 1
date: 19 December 2023
bibliography: paper.bib
---

# Summary

PyCVI is a Python package specialized in internal Clustering Validity Indices (CVI) compatible with both time-series and non time-series data.

Clustering is a task that aims at finding groups within a given dataset. CVIs are used to select the best clustering among a pre-computed set of clusterings. In other words, CVIs select the division of the dataset into groups that best ensures that similar datapoints belong to the same group and non-related datapoints are in different groups. *Internal* CVIs are essential in practice when no *external* information is available about the dataset such as the true association of the datapoints with groups.

External CVIs are however great tools to evaluate the performance of clustering methods and internal CVIs on benchmark datasets [@Gurrutxaga2011]. PyCVI then also implements the Variation of Information for that purpose.

# Statement of need

Although being fundamental to clustering tasks and being an active research topic, very few internal CVIs are implemented in standard python libraries (only 3 in [scikit-learn](https://scikit-learn.org/stable/index.html), more were available in R but few were maintained and kept in CRAN). This is despite the well-known limitations of all existing CVIs and the need to use the right one(s) according to the specific dataset at hand.

In addition, all CVIs rely on the definition of a distance between datapoints and most of them on the notion of cluster center.

For non-time-series data, the distance used is usually the euclidean distance and the cluster center is defined as the usual average. Libraries such as [scipy](https://docs.scipy.org/doc/scipy/index.html), [numpy](https://numpy.org/doc/stable/), [scikit-learn](https://scikit-learn.org/stable/index.html), etc. offer a large selection of distance measures that are compatible with all their functions.

For time-series data however, the common distance used is Dynamic Time Warping (DTW) and the barycenter of a group of time series is then not defined as the usual mean, but as the DTW Barycentric Average (DBA). Unfortunately, DTW and DBA are not compatible with the libraries mentioned above.

PyCVI then tries to fill that gap by implementing 11 state-of-the-art internal CVIs: Hartigan, Calinski-Harabasz, GapStatistic, Silhouette, ScoreFunction, Maulik-Bandyopadhyay, SD, SDbw, Dunn, Xie-Beni and Davies-Bouldin. Then, in PyCVI their definition is extended in order to make them compatible with DTW and DBA in addition to non time-series data. PyCVI is entirely compatible with [scikit-learn](https://scikit-learn.org/stable/index.html), [scikit-learn-extra](https://scikit-learn-extra.readthedocs.io/en/stable/), [aeon](https://www.aeon-toolkit.org/en/latest/index.html) and [sktime](https://www.sktime.net/en/stable/index.html), in order to be easily integrated into any clustering pipeline in python.

# Example

We experimented on 3 different cases: non time-series data, time-series data with euclidean distance and time-series data with DTW and DBA as distance measure and center of clusters. In addition, for each case, we used a clustering method from a different library: KMeans and AgglomerativeClustering from sklearn, TimeSeriesKMeans from aeon and KMedoids from scikit-learn-extra in order to give examples of integration with other clustering libraries. Then, for each case, we ran all the CVIs implemented in PyCVI and selected the best clustering according to each CVI. Finally we computed the variation of information between each selected clustering and the true clustering. A heavy variation of information, illustrate a poor clustering quality, but this can be due to the clustering method and not the CVI as illustrated by the second plot of each figure which shows that even when assuming the correct number of clusters, a clustering method can fail at clustering the dataset.

The code of this example is available on the GitHub repository of the package, as well as on its documentation.

# Acknowledgements

We thank the climate and energy transition strategy of the University of Bergen in Norway (UiB) for funding this work.

# References
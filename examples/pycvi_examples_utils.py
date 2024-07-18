import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from typing import List, Tuple, Dict, Any
from math import ceil

from pycvi.cvi import CVIs
from pycvi.cluster import get_clustering


def _get_nrows_ncols(nplots: int = None):
    """
    Adapt the figures to the total number of CVIs.

    We want to know before creating the figure how many rows and columns
    we will need. This depends on how many different k will be selected
    and potentially if we have additional plots for the true data.

    Parameters
    ----------
    nplots : int, optional
        Number of plots, by default None, resulting in `nplots=len(CVIs)
        + 2`
    """
    if nplots is None:
        nplots = len(CVIs) + 2
    n_rows = ceil(nplots / 5)
    n_cols = 5
    figsize = (4*n_cols, ceil(4*n_rows))
    return n_rows, n_cols, figsize

def _get_shape_UCR(data: np.ndarray) -> Tuple[Tuple[int], bool]:
    """
    Get the shape (N, T, d) of data and whether it is time series data.

    Parameters
    ----------
    data : np.ndarray
        The original data

    Returns
    -------
    Tuple[Tuple[int], bool]
        the shape (N, T, d) and whether it is time series data
    """
    dims = data.shape
    if len(dims) == 3:
        (N, T, d) = data.shape
        UCR = True
    else:
        (N, d) = data.shape
        T = 1
        UCR = False
    return (N, T, d), UCR

def _get_colors(name: str="Set1") -> List:
    """
    Helper function to get a list of colors

    Parameters
    ----------
    name : str, optional
        Name of the matplotlib cmap, by default "Set1".

    Returns
    -------
    List
        A list of colors
    """
    cmap = get_cmap(name)
    colors = cmap.colors
    return colors

def plot_cluster(
    ax,
    data: np.ndarray,
    cluster: List[int],
    color,
):
    """
    Plot a given cluster on the given ax.

    Works with UCR data (plot lines), and with non-time series data with
    dimensions d = 1,2 or 3.

    In case it is UCR data, use "color" for each line representing a
    datapoint in the cluster.

    Parameters
    ----------
    ax : A matplotlib axes
        Where to plot the cluster
    data : np.ndarray
        The dataset
    cluster : List[int]
        The indices representing the cluster
    color : _type_
        The color to use to plot the cluster

    Returns
    -------
    A matplotlib axes
        The same matplotlib axes, but with the cluster plotted.
    """
    # Get the full shape and whether it is time-series data.
    (N, T, d), UCR = _get_shape_UCR(data)

    # If UCR, use plot type of plots.
    if UCR:
        # Transparency
        alpha = 0.2
        x = np.arange(T)
        y_val = data[cluster, :, 0]

        # Plot lines one by one, with the same color.
        for y in y_val:
            ax.plot(x, y, c=color, alpha=alpha)

    # If non time series data, use scatter plots.
    else:
        # Size of the dots
        s = 2
        if d == 1:
            x_val = np.zeros_like(data[cluster, 0])
            y_val = data[cluster, 0]
            ax.scatter(x_val, y_val, s=s)
        elif d == 2:
            x_val = data[cluster, 0]
            y_val = data[cluster, 1]
            ax.scatter(x_val, y_val, s=s)
        elif d == 3:
            x_val = data[cluster, 0]
            y_val = data[cluster, 1]
            z_val = data[cluster, 2]
            ax.scatter(x_val, y_val, z_val, s=s)

    return ax

def plot_center(
    ax,
    data: np.ndarray,
    center,
    color,
):
    """
    Plot a given cluster center on the given ax.

    Works with UCR data (plot lines), and with non-time series data with
    dimensions d = 1,2 or 3.

    Parameters
    ----------
    ax : A matplotlib axes
        Where to plot the cluster
    data : np.ndarray
        The dataset
    center :
        The center of a cluster
    color : _type_
        The color to use to plot the cluster

    Returns
    -------
    A matplotlib axes
        The same matplotlib axes, but with the cluster and centers
        plotted.
    """
    # Get the full shape and whether it is time-series data.
    (N, T, d), UCR = _get_shape_UCR(data)

    # If UCR, use plot type of plots.
    if UCR:
        # Transparency
        alpha = 1
        x = np.arange(T)
        y_val = center
        ax.plot(x, y_val, c=color, alpha=alpha)

    # If non time series data, use scatter plots.
    else:
        # Size of the dots
        s = 10
        if d == 1:
            x_val = np.zeros_like(center)
            y_val = center
            ax.scatter(x_val, y_val, s=s)
        elif d == 2:
            x_val = center[0]
            y_val = center[1]
            ax.scatter(x_val, y_val, s=s)
        elif d == 3:
            x_val = center[0, 0]
            y_val = center[0, 1]
            z_val = center[0, 2]
            ax.scatter(x_val, y_val, z_val, s=s)

    return ax

def plot_hist_selected(
    summary_selected: Dict[int, Dict[str, Any]],
):
    """
    Bar plot of number of CVIs that selected a given number of clusters.

    :param summary_selected: A dictionary containing for each selected k
        ("k_selected"), all information on the selected clustering
        ("#CVI, "clustering", "ax_title")
    :type summary_selected: List[List[List[int]]]
    :return: a figure with one clustering per CVI (+2 plots first)
    :rtype: A matplotlib figure
    """

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(5, 5), tight_layout=True
    )
    n_CVIs = [
        summary["#CVI"] for summary in summary_selected.values()
    ]
    # Plot historgram
    ax.bar(summary_selected.keys(), n_CVIs)

    # Force x_ticks to be int and to appear also for 0 values
    k_min, k_max = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(k_min, k_max, 1, dtype=int))

    # Labels and titles
    ax.set_xlabel("Selected number of clusters")
    ax.set_ylabel("Number of CVIs that selected this number of clusters")
    fig.suptitle("Number of CVIs that selected a given number of clusters")

    return fig

def plot_only_selected(
    data: np.ndarray,
    summary_selected: Dict[int, Dict[str, Any]],
    fig,
):
    """
    Add one plot per selected number of clusters.

    Parameters
    ----------
    data : np.ndarray, shape (N, d)
        Original data, corresponding to a benchmark dataset
    summary_selected : Dict[int, Dict[str, Any]]
        A dictionary containing for each selected k ("k_selected"), all
        information on the selected clustering ("#CVI, "clustering",
        "ax_title")
    fig : A matplotlib figure
        Figure where all the plots are (including 2 about the true
        clusters)

    Returns
    -------
    A matplotlib figure
        A figure with one clustering per CVI (+2 plots first)
    """

    colors = _get_colors()

    # -------  Plot the clustering selected by a given CVI -----------

    # Sort the dictionnary by number of CVI that selected k
    sorted_summary = {
        k: n_cvi for k, n_cvi
        in sorted(
            summary_selected.items(), key=lambda d: d[1]["#CVI"], reverse=True
        )
    }
    for i, (k, summary) in enumerate(summary_selected.items()):

        # Find the ax corresponding to the CVI
        ax = fig.axes[i+2] # i+2 because there are 2 plots already

        # Add predefined title
        ax.set_title(str(summary["ax_title"]))

        # ------------------ Plot clusters one by one ------------------
        for i_label, cluster in enumerate(summary["clustering"]):
            color = colors[i_label % len(colors)]
            ax = plot_cluster(ax, data, cluster, color)

    # Remove empty axes
    for ax in fig.axes[len(summary_selected)+2:]:
        ax.remove()

    return fig

def plot_selected_clusters(
    data: np.ndarray,
    clusterings_selected: List[List[List[int]]],
    fig,
    titles: List[str],
):
    """
    Add one plot per CVI with their corresponding selected clustering.

    The fig should already contain 2 plots first with the true
    clusterings, and the clusterings obtained with k_true.

    :param data: Original data, corresponding to a benchmark dataset
    :type data: np.ndarray, shape (N, d)
    :param clusterings_selected: A list of n_CVI clusterings.
    :type clusterings_selected: List[List[List[int]]]
    :param fig: Figure where all the plots are (including 2 about the
        true clusters)
    :type fig:
    :param titles: List of titles for each CVI
    :type titles: List[str]
    :return: a figure with one clustering per CVI (+2 plots first)
    :rtype: A matplotlib figure
    """

    colors = _get_colors()

    # -------  Plot the clustering selected by a given CVI -----------
    for i_CVI in range(len(clusterings_selected)):

        # Find the ax corresponding to the CVI
        ax = fig.axes[i_CVI+2] # i+2 because there are 2 plots already

        # Add predefined title
        ax.set_title(str(titles[i_CVI]))
        if clusterings_selected[i_CVI] is None:
            continue

        # ------------------ Plot clusters one by one ------------------
        for i_label, cluster in enumerate(clusterings_selected[i_CVI]):
            color = colors[i_label % len(colors)]
            ax = plot_cluster(ax, data, cluster, color)

    # Remove empty axes
    for ax in fig.axes[len(clusterings_selected)+2:]:
        ax.remove()

    return fig

def plot_true_selected(
    data: np.ndarray,
    clustering_true: List[List[int]],
    clustering_pred: List[List[int]],
    ax_titles: List[str] = None,
):
    """
    Plot the true clustering and the selected clustering.

    :param ax_titles: List of titles for the two plots
    :type ax_titles: List[str]
    :return: a figure with one clustering per CVI (+2 plots first)
    :rtype: A matplotlib figure

    Parameters
    ----------
    data : np.ndarray, shape `(N, d)`
        Original data, corresponding to a benchmark dataset
    clustering_true : List[List[int]]
        True clustering
    clustering_pred : List[List[int]]
        Predicted clustering
    ax_titles : List[str], optional
        List of titles for the two plots, by default None

    Returns
    -------
    A matplotlib figure
        A figure with the true clustering and the selected clustering.
    """
    (N, T, d), UCR = _get_shape_UCR(data)
    colors = _get_colors()

    # ----------------------- Create figure ----------------
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(10, 5), tight_layout=True
        )

    # ------------------- variables for the 2 axes ----------------
    clusters = [
        clustering_true,
        clustering_pred,
    ]
    if ax_titles is None:
        ax_titles = [
            f"True labels, k={len(clustering_true)}",
            f"Clustering selected, with k={len(clustering_pred)}",
        ]

    # ----  Plot the true clustering and the clustering selected  ------
    for i_ax in range(len(clusters)):

        ax = fig.axes[i_ax]
        ax.set_title(str(ax_titles[i_ax]))

        # ------------------ Plot clusters one by one ------------------
        for i_clus, cluster in enumerate(clusters[i_ax]):
            color = colors[i_clus % len(colors)]
            ax = plot_cluster(ax, data, cluster, color)
    return fig

def plot_true_best(
    data: np.ndarray,
    labels: np.ndarray,
    clusterings: List[List[List[int]]],
    VI_best: float = None,
    n_plots: int = None
):
    """
    Plot the true clustering and the clustering obtained with k_true.

    Create also the whole figure that will be used to plot the
    clusterings selected by each CVI.

    Parameters
    ----------
    data : np.ndarray, shape (N, d)
        Original data, corresponding to a benchmark dataset
    labels : np.ndarray, shape (N,)
        True labels
    clusterings : List[List[List[int]]]
        The clusterings obtained with k_true
    VI_best : float, optional
        The VI between the true clustering and the clustering assuming
        the right number of clusters., by default None
    n_plots : int, optional
        Number of plots to add after the two initial plots, by default
        None

    Returns
    -------
    A matplotlib figure
        The figure with 2 plots on it, and many empty axes.
    """

    (N, T, d), UCR = _get_shape_UCR(data)
    colors = _get_colors()

    # ----------------------- Create figure ----------------
    if d <= 2:
        nrows, ncols, figsize = _get_nrows_ncols(n_plots)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=True, sharey=True,
            figsize=figsize, tight_layout=True
        )
    else:
        return None

    # ----------------------- Labels ----------------
    if labels is None:
        labels = np.zeros(N)
    classes = np.unique(labels)
    n_labels = len(classes)
    if n_labels == N:
        labels = np.zeros(N)
        n_labels = 1

    # ------------------- variables for the 2 axes ----------------
    clusters = [
        # The true clustering
        [labels == classes[i] for i in range(n_labels)],
        # The clustering obtained with k_true
        clusterings
    ]
    if VI_best is not None:
        ax_titles = [
            f"True labels, k={n_labels}",
            f"Clustering assuming k={n_labels} | VI={VI_best:.4f}",
        ]
    else:
        ax_titles = [
            f"True labels, k={n_labels}",
            f"Clustering assuming k={n_labels}",
        ]

    # ------ True clustering and clustering assuming n_labels ----------
    for i_ax in range(2):
        if d <= 2:
            ax = fig.axes[i_ax]

        # ---------------  Plot clusters one by one --------------------
        for i_label in range(n_labels):
            c = clusters[i_ax][i_label]
            color = colors[i_label % len(colors)]
            ax = plot_cluster(ax, data, c, color)

        # Add title
        ax.set_title(ax_titles[i_ax])

    return fig

def plot_centers(
    data: np.ndarray,
    clustering: List[List[int]],
    cluster_centers: List,
):
    """
    Plot the clustering and their cluster centers

    Parameters
    ----------
    data : np.ndarray
        The data.
    clustering : List[List[int]]
        The labels.
    cluster_centers : List
        The cluster centers of the clusters.

    Returns
    -------
    A matplotlib figure
        A figure with 2 plots: the clustering and the cluster centers
    """
    (N, T, d), UCR = _get_shape_UCR(data)
    colors = _get_colors()

    # ----------------------- Create figure ----------------
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5),
        tight_layout=True
    )

    ax_titles = [
        f"Clustering",
        f"Cluster centers",
    ]

    ax = fig.axes[0]
    ax.set_title(ax_titles[0])
    for i, cluster in enumerate(clustering):
        ax = plot_cluster(ax, data, cluster, colors[i])


    ax = fig.axes[1]
    ax.set_title(ax_titles[1])
    for i, center in enumerate(cluster_centers):
        ax = plot_center(ax, data, center, colors[i])

    return fig
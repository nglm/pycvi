import numpy as np
import matplotlib.pyplot as plt
from typing import List
from math import ceil

from pycvi.cvi import CVIs

# --------  Adapt the figures to the total number of scores ------------
N_CVIs = len(CVIs)
N_ROWS = ceil(N_CVIs / 5)
N_COLS = 5
FIGSIZE = (4*N_COLS, ceil(2.5*N_ROWS))
# ----------------------------------------------------------------------

def plot_clusters(
    data: np.ndarray,
    clusterings_selected: List[List[List[int]]],
    fig,
    titles: List[str],
):
    """
    Add one plot per score with their corresponding selected clustering

    The fig should already contain 2 plots first with the true
    clusterings, and the clusterings obtained with k_true.

    :param data: Original data, corresponding to a benchmark dataset
    :type data: np.ndarray, shape (N, d)
    :param clusterings_selected: A list of n_score clusterings.
    :type clusterings_selected: List[List[List[int]]]
    :param fig: Figure where all the plots are (including 2 about the
        true clusters)
    :type fig:
    :param titles: List of titles for each score
    :type titles: List[str]
    :return: a figure with one clustering per score (+2 plots first)
    :rtype:
    """
    (N, d) = data.shape
    # -------  Plot the clustering selected by a given score -----------
    for i_score in range(len(clusterings_selected)):

        # ------------- Find the ax corresponding to the score ---------
        if d <= 2:
            ax = fig.axes[i_score+2] # i+2 because there are 2 plots already
        else:
            return None

        # Add predefined title
        ax.set_title(str(titles[i_score]))

        # ------------------ Plot clusters one by one ------------------
        for i_label, cluster in enumerate(clusterings_selected[i_score]):
            if d == 1:
                ax.scatter(np.zeros_like(data[cluster, 0]), data[cluster, 0], s=1)
            elif d == 2:
                ax.scatter(data[cluster, 0], data[cluster, 1], s=1)

    return fig



def plot_true(
    data: np.ndarray,
    labels: np.ndarray,
    clusterings: List[List[List[int]]],
):
    """
    Plot the true clustering and the clustering obtained with k_true

    Create also the whole figure that will be used to plot the
    clusterings selected by each score.

    :param data: Original data, corresponding to a benchmark dataset
    :type data: np.ndarray, shape (N, d)
    :param labels: True labels
    :type labels: np.ndarray
    :param clusterings: The clusterings obtained with k_true
    :type clusterings: List[List[List[int]]]
    :return: _description_
    :rtype: _type_
    """

    (N, d) = data.shape

    # ----------------------- Create figure ----------------
    if d <= 2:
        fig, axes = plt.subplots(
            nrows=N_ROWS, ncols=N_COLS, sharex=True, sharey=True,
            figsize=FIGSIZE, tight_layout=True
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
    ax_titles = [
        "True labels, k={}".format(n_labels),
        "Clustering assuming k={}".format(n_labels),
    ]

    # ------ True clustering and clustering assuming n_labels ----------
    for i_ax in range(2):
        if d <= 2:
            ax = fig.axes[i_ax]

        # ---------------  Plot clusters one by one --------------------
        for i in range(n_labels):

            c = clusters[i_ax][i]
            if d == 1:
                ax.scatter(
                    np.zeros_like(data[c, 0]), data[c, 0], s=1
                )
            elif d == 2:
                ax.scatter(data[c, 0], data[c, 1], s=1)

        # Add title
        ax.set_title(ax_titles[i_ax])

    return fig
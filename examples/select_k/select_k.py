import sys
sys.path.append('./examples')
out_fname = f'./output-select_k.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout

import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from aeon.clustering import TimeSeriesKMeans

from pycvi.cluster import generate_all_clusterings, get_clustering
from pycvi.cvi import CVIs
from pycvi.compute_scores import compute_all_scores
from pycvi.vi import variation_information
from pycvi.datasets.benchmark import load_data
from pycvi.exceptions import SelectionError

from pycvi_examples_utils import plot_true_best, plot_hist_selected, plot_only_selected

def pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kw: dict,
    k_max: int = 25,
    scaler = StandardScaler(),
    DTW:bool = False,
    fig_title: str = "",
    fig_name: str = "",
) -> None:
    """
    This function gives an example of typical pipeline using PyCVI.

    In this example we assume that we are in real conditions, which
    means that we don't have access to the true labels (except for the
    final figure). We then don't use the features included in the
    :mod:`pycvi.vi` module. In this function we:

    - Standardize the data
    - Generate all clusterings for a given range of number of clusters.
    - For each CVI available in PyCVI, compute their values for all
      generated clusterings
    - For each CVI, select the best clustering according to its CVI
      values
    - Create a summary plot containing the true clustering, the
      clustering assuming the correct number of clusters, for each
      potential number of clusters :math:`k`, the number of CVI that
      selected :math:`k` and the selected clusterings of each CVI.
    """
    print(f'\n ***** {fig_title} ***** \n')
    k_range = range(k_max)

    # ------------------------------------------------------------------
    # ------------------ Define true clustering  -----------------------
    # ------------------------------------------------------------------
    # From the label for each datapoint to a list of
    # datapoints for each cluste.
    # true clusters: List[List[int]]
    true_clusters = get_clustering(y)
    k_true = len(true_clusters)

    # ------------------------------------------------------------------
    # ------------------ Generate clusterings  -------------------------
    # ------------------------------------------------------------------

    t_start = time.time()

    clusterings = generate_all_clusterings(
            X,
            model_class,
            model_kw=model_kw,
            n_clusters_range = k_range,
            DTW = DTW,
            scaler=scaler,
        )

    t_end = time.time()
    dt = t_end - t_start

    print(f"Clusterings generated in: {dt:.2f}s")

    # ------------------------------------------------------------------
    # ------------ Compute CVI values and select k ---------------------
    # ------------------------------------------------------------------
    summary = {}

    for cvi_class in CVIs:

        # Instantiate a CVI model
        cvi = cvi_class()
        t_start = time.time()
        print(f" ====== {cvi} ====== ")

        # Compute CVI values for all clusterings
        scores = compute_all_scores(
            cvi,
            X,
            clusterings,
            DTW=DTW,
            scaler=StandardScaler(),
        )

        t_end = time.time()
        dt = t_end - t_start

        # Print CVI values and selected k
        for k in clusterings:
            print(k, scores[k])
        print('Code executed in %.2f s' %(dt))

        # Select k
        try:
            k_selected = cvi.select(scores)
        # If no k could be selected with this CVI don't do anything...
        except SelectionError as e:
            k_selected = None

        # Otherwise update summary info related to the selected clusterings
        else:
            if k_selected not in summary:
                summary[k_selected] = {}
            summary[k_selected]["clustering"] = clusterings[k_selected]
            summary[k_selected]["#CVI"] = summary[k_selected].pop("#CVI", 0) + 1
            ax_title = f'k={k_selected}, #CVI={summary[k_selected]["#CVI"]}'
            summary[k_selected]["ax_title"] = ax_title
        finally:
            print(f"Selected k: {k_selected} | True k: {k_true}")

    # ------------------------------------------------------------------
    # --------------------- Plot true clustering -----------------------
    # ------------------------------------------------------------------

    best_clusters = clusterings[k_true]

    # -- Plot true clusters & clusters when assuming k_true clusters ---
    fig = plot_true_best(X, y, best_clusters, n_plots=len(summary) + 2)

    # ------------------------------------------------------------------
    # ----------------------- Summary plot -----------------------------
    # ------------------------------------------------------------------

    fig_hist = plot_hist_selected(summary)
    fig_hist.savefig(fig_name + "-histogram.png")

    fig = plot_only_selected(X, summary, fig)
    fig.suptitle(fig_title)
    fig.savefig(fig_name + ".png")

# ======================================================================
# PyCVI on non time-series data
# ======================================================================

# ------------- KMeans ------------------------
X, y = load_data("zelnik1", "barton")
DTW = False
k_max = 10

model_class = KMeans
model_kw = {}
scaler = StandardScaler()

fig_title = "Non time-series data with KMeans"
fig_name = "Barton_data_KMeans"
pipeline(X, y, model_class, model_kw, k_max, scaler, DTW, fig_title, fig_name)

# --------- AgglomerativeClustering ----------
X, y = load_data("zelnik1", "barton")
DTW = False
k_max = 10

model_class = AgglomerativeClustering
# sklearn kwargs for AgglomerativeClustering
model_kw = {"linkage" : "single"}
scaler = StandardScaler()

fig_title = "Non time-series data with AgglomerativeClustering-Single"
fig_name = "Barton_data_AgglomerativeClustering_Single"

pipeline(X, y, model_class, model_kw, k_max, scaler, DTW, fig_title, fig_name)

# ======================================================================
# PyCVI on time series data
# ======================================================================

X, y = load_data("Trace", "UCR")

# ==========================
# PyCVI using DTW
# ==========================

DTW = True

model_class = TimeSeriesKMeans
# aeon kwargs for TimeSeriesKMeans
model_kw = {
    "distance" : "dtw",
    "distance_params" : {"window": 0.2},
}
scaler = StandardScaler()
fig_title = "Time-series data using DTW with TimeSeriesKMeans"
fig_name = "UCR_data_DTW_TimeSeriesKMeans"

pipeline(X, y, model_class, model_kw, k_max, scaler, DTW, fig_title, fig_name)

# ==========================
# PyCVI not using DTW
# ==========================

DTW = False

model_class = KMedoids
model_kw = {}
scaler = StandardScaler()
fig_title = "Time-series data without DTW with KMedoids"
fig_name = "UCR_data_no_DTW_KMedoids"

pipeline(X, y, model_class, model_kw, k_max, scaler, DTW, fig_title, fig_name)

fout.close()
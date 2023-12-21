import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from aeon.clustering import TimeSeriesKMeans
import time
import sys

from pycvi.cluster import generate_all_clusterings
from pycvi.cvi import CVIs
from pycvi.compute_scores import compute_all_scores
from pycvi.vi import variation_information
from pycvi.datasets.benchmark import load_data

from .utils import plot_true, plot_clusters

out_fname = f'./output-full_example.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout

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

    In this function we:

    - Generate all clusterings for a given range of number of clusters.
    - Compute the Variation of Information (VI) between the generated
      clusterings and the true clustering
    - For each CVI available in PyCVI, compute their values for all
      generated clusterings
    - For each CVI, select the best clustering according to its CVI
      values
    - Create a summary plot containing the true clustering, the
      clustering assuming the correct number of clusters and the
      selected clusterings of each CVI.
    """
    print(f'\n ***** {fig_title} ***** \n')
    N = len(X)
    k_range = range(min(k_max, N+1))

    # ------------------------------------------------------------------
    # ------------------ Define true clustering  -----------------------
    # ------------------------------------------------------------------
    classes = np.unique(y)
    k_true = len(classes)

    # true clusters: List[List[int]]
    indices = np.arange(len(X))
    true_clusters = [ indices[y == classes[i]] for i in range(k_true) ]

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
        )[0]

    t_end = time.time()
    dt = t_end - t_start

    print(f"Clusterings generated in: {dt:.2f}s")

    # ------------------------------------------------------------------
    # ------  Variation of information with the true clustering --------
    # ------------------------------------------------------------------

    best_clusters = clusterings[k_true]
    VI_best = variation_information(true_clusters, best_clusters)

    # -- Plot true clusters & clusters when assuming k_true clusters ---
    fig = plot_true(X, y, best_clusters, VI_best)

    print(" ================ VI ================ ")
    # Compute VI between the true clustering and each clustering
    # obtained with the different clustering methods
    VIs = {}
    for k, clustering in clusterings.items():
        VIs[k] = variation_information(true_clusters, clustering)
        print(k, VIs[k])

    # ------------------------------------------------------------------
    # ------------ Compute CVI values and select k ---------------------
    # ------------------------------------------------------------------
    ax_titles = []
    clusterings_selected = []

    for cvi_class in CVIs:

        # Instantiate a CVI model
        cvi = cvi_class()
        t_start = time.time()
        print(f" ====== {cvi} ====== ")

        # Compute CVI values for all clusterings
        scores = compute_all_scores(
            cvi,
            X,
            [clusterings],
            DTW=DTW,
            scaler=StandardScaler(),
        )

        t_end = time.time()
        dt = t_end - t_start

        # Select k
        k_selected = cvi.select(scores)[0]

        # Print CVI values and selected k
        for k in clusterings:
            print(k, scores[0][k])
        print(f"Selected k: {k_selected} | True k: {k_true}")
        print('Code executed in %.2f s' %(dt))

        # For plotting purpose
        clusterings_selected.append(clusterings[k_selected])
        ax_titles.append(
            f"{cvi}, k={k_selected}, VI={VIs[k_selected]:.2f}"
        )

    # ------------------------------------------------------------------
    # ----------------------- Summary plot -----------------------------
    # ------------------------------------------------------------------

    fig = plot_clusters(X, clusterings_selected, fig, ax_titles)

    fig.suptitle(fig_title)
    fig.savefig(fig_name + ".png")

# ======================================================================
# PyCVI on non time-series data
# ======================================================================

# long1
# zelnik1

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
model_kw = {"linkage" : "single"}
scaler = StandardScaler()

fig_title = "Non time-series data with AgglomerativeClustering-Single"
fig_name = "Barton_data_AgglomerativeClustering_Single"

pipeline(X, y, model_class, model_kw, k_max, scaler, DTW, fig_title, fig_name)

# ======================================================================
# PyCVI on time series data
# ======================================================================

# Trace dataset
# SmallKitchenAppliances dataset
X, y = load_data("Trace", "UCR")

# ==========================
# PyCVI using DTW
# ==========================

DTW = True

model_class = TimeSeriesKMeans
model_kw = {}
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
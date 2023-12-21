from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering, compute_center

from ..utils import plot_centers

# ===================== Non time series-data ===========================

data, labels = load_data("xclara", "barton")

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.clustering_true = get_clustering(labels)
clustering_true = get_clustering(labels)

# -------------------- Compute cluster centers -------------------------
cluster_centers = [compute_center(data[c]) for c in clustering_true]

# ------------------------ Summary plot --------------------------------
fig = plot_centers(data, clustering_true, cluster_centers)
fig_title = "Non time-series data - xclara"
fig_name = "cluster_centers.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)

# ======================= Time series-data =============================

data, labels = load_data("Trace", "ucr")

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.clustering_true = get_clustering(labels)
clustering_true = get_clustering(labels)

# -------------------- Compute cluster centers -------------------------
cluster_centers = [compute_center(data[c]) for c in clustering_true]

# ------------------------ Summary plot --------------------------------
fig = plot_centers(data, clustering_true, cluster_centers)
fig_title = "Time-series data - Trace"
fig_name = "cluster_centers_TS.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)
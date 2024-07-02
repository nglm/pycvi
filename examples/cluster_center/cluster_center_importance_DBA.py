import sys
sys.path.append('./examples')

from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering, compute_center

from pycvi_examples_utils import plot_centers

# ======================= Time series-data =============================

data, labels = load_data("Trace", "ucr")

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.clustering_true = get_clustering(labels)
clustering_true = get_clustering(labels)

# ========================== Using DBA =================================
# -------------------- Compute cluster centers -------------------------
cluster_centers = [compute_center(data[c]) for c in clustering_true]

# ------------------------ Summary plot --------------------------------
fig = plot_centers(data, clustering_true, cluster_centers)
fig_title = "Time-series data - Trace - Using DBA"
fig_name = "cluster_centers_TS_DBA.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)

# ========================= Without DBA ================================
# -------------------- Compute cluster centers -------------------------
(N, T, d) = data.shape
cluster_centers = [
    compute_center(data[c].reshape(len(data[c]), -1)).reshape(T,d)
    for c in clustering_true
]
# ------------------------ Summary plot --------------------------------
fig = plot_centers(data, clustering_true, cluster_centers)
fig_title = "Time-series data - Trace - Using Euclidean mean"
fig_name = "cluster_centers_TS_without_DBA.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)
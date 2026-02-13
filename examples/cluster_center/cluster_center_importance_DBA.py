import sys
sys.path.append('./examples')

from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering, compute_centers

from pycvi_examples_utils import plot_centers

# ======================= Time series-data =============================

data, labels = load_data("Trace", "ucr")

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.clustering_true = get_clustering(labels)
clustering_true = get_clustering(labels)

# ========================== Using DBA =================================
# -------------------- Compute cluster centers -------------------------
cluster_centers = compute_centers(data, clustering_true)

# ------------------------ Summary plot --------------------------------
fig = plot_centers(data, clustering_true, cluster_centers)
fig_title = "Time-series data - Trace - Using DBA"
fig_name = "cluster_centers_TS_DBA.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)

# ========================= Without DBA ================================
# -------------------- Compute cluster centers -------------------------
# Consider the data as static data even if they are time series
(N, T, d) = data.shape
data = data.reshape(N, T * d)

cluster_centers = compute_centers(data, clustering_true)

# ------------------------ Summary plot --------------------------------

# Condiser the cluster centers as time-series data again for plotting
cluster_centers = [center.reshape(T, d) for center in cluster_centers]
fig = plot_centers(data, clustering_true, cluster_centers)
fig_title = "Time-series data - Trace - Using Euclidean mean"
fig_name = "cluster_centers_TS_without_DBA.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)
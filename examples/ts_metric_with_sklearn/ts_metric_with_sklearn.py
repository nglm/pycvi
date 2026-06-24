from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering
from pycvi.dist import time_series_metric_with_sklearn

from pycvi_examples_utils import plot_true_selected

# -------------- Standard data handling operations ---------------------
# Load data
data, labels = load_data("Trace", "ucr")
(N, T, d) = data.shape

# Data pre-processing
scaler = StandardScaler()

# Scaling for each variable and not time step wise
X = scaler.fit_transform(data.reshape(N*T, d)).reshape(N, T, d)

# Reshape data to match sklearn requirements
X = data.reshape(N, T*d)

# ---------- Fit a clustering model and make predictions ---------------
# Assumed number of clusters
k = 4

# Train and predict a AgglomerativeClustering model with a Time-series metric
model = AgglomerativeClustering(
    n_clusters=k,
    metric=time_series_metric_with_sklearn(X, d=d, T=T),
    linkage="single",
)

labels_pred = model.fit_predict(X)

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.
clusters_pred = get_clustering(labels_pred)

# ---------------------- Summmary fig ----------------------------------

clustering_true = get_clustering(labels)
fig = plot_true_selected(data, clustering_true, clusters_pred)
fig_title = "AgglomerativeClustering with time-series distance"
fig_name = "ts_metric_with_sklearn_Agglo.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)
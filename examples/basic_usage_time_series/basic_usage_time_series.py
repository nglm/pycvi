import sys
out_fname = f'./output-basic_usage_TS_KMeans_Dunn.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout


from aeon.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
from pycvi.cvi import Dunn
from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering

from ..utils import plot_true_selected

# -------------- Standard data handling operations ---------------------
# Load data
data, labels = load_data("Trace", "ucr")
(N, T, d) = data.shape

# Data pre-processing
scaler = MinMaxScaler()
# Scaling for each variable and not time step wise
X = scaler.fit_transform(data.reshape(N*T, d)).reshape(N, T, d)
# CVI to use, could be any class defined in pycvi.cvi
cvi = Dunn()

# ---------- Integrating PyCVI in the clustering pipeline --------------
# ------ 1. Compute CVI values of the generated clusterings ------------
# ------ 2. Select the best clustering according to the CVI ------------

clusterings = {}
cvi_values = {}
k_range = range(2, 10)
for k in k_range:

    # Generate the clusters assuming that there are k clusters
    # Clustering model to use, could be any sklearn-like clustering class
    model = TimeSeriesKMeans(n_clusters=k)
    labels_pred = model.fit_predict(X)

    # From predicted cluster-label for each datapoint to a list of
    # datapoints for each cluster.
    clusters_pred = get_clustering(labels_pred)

    # Compute the CVI value of this clustering
    cvi_value = cvi(X, clusters_pred)

    # Store clustering and CVI value
    clusterings[k] = clusters_pred
    cvi_values[k] = cvi_value
    print(f"k={k}  |  CVI value:{cvi_value}")

k_selected = cvi.select(cvi_values)
print(f"k selected: {k_selected}")

# ---------------------- Summmary fig ----------------------------------

clustering_true = get_clustering(labels)
fig = plot_true_selected(data, clustering_true, clusterings[k_selected])
fig_title = "KMeans clustering with Dunn score"
fig_name = "basic_usage_TS_KMeans_Dunn.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)

fout.close()
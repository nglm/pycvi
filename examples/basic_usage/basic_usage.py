import sys
out_fname = f'./output-basic_usage_KMeans_Silhouette.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pycvi.cvi import Silhouette
from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering

from ..utils import plot_true_selected

# -------------- Standard data handling operations ---------------------
# Load data
data, labels = load_data("xclara", "barton")

# Data pre-processing
scaler = StandardScaler()
X = scaler.fit_transform(data)
# CVI to use, could be any class defined in pycvi.cvi
cvi = Silhouette()

# ---------- Integrating PyCVI in the clustering pipeline --------------
# ------ 1. Compute CVI values of the generated clusterings ------------
# ------ 2. Select the best clustering according to the CVI ------------

clusterings = {}
cvi_values = {}
k_range = range(2, 10)
for k in k_range:

    # Generate the clusters assuming that there are k clusters
    # Clustering model to use, could be any sklearn-like clustering class
    model = KMeans(n_clusters=k)
    labels_pred = model.fit_predict(X)

    # From predicted cluster-label for each datapoint to a list of
    # datapoint for each cluster.
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
fig_title = "KMeans clustering with Silhouette score"
fig_name = "basic_usage_KMeans_Silhouette.png"
fig.suptitle(fig_title)
fig.savefig(fig_name)

fout.close()
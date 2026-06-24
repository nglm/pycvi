import sys
out_fname = f'./output-ts_metric_with_sklearn.txt'
fout = open(out_fname, 'wt')
sys.path.append('./examples')
sys.stdout = fout

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from pycvi.cvi import CalinskiHarabasz
from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering
from pycvi.dist import time_series_metric_with_sklearn

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

# CVI to use, could be any class defined in pycvi.cvi
cvi = CalinskiHarabasz()

# ---------- Fit a clustering model and make predictions ---------------
# Assumed number of clusters
k = 4

# Train and predict a AgglomerativeClustering model
model = AgglomerativeClustering(n_clusters=k, metric=time_series_metric_with_sklearn(X,))
labels_pred = model.fit_predict(X)

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.
clusters_pred = get_clustering(labels_pred)

# ---------------- Using Object-oriented API -----------------------
# Instanciate a CVI instance, could be any class defined in pycvi.cvi
cvi = Silhouette()
cvi_kwargs = {"dist_kwargs": {"metric": "minkowski", "p": 3}}
cvi_value = cvi(X, clusters_pred, cvi_kwargs=cvi_kwargs)
print(f"OOP API           |  CVI value: {cvi_value:.4f}")

# ---------------- Using Functional API -----------------------
dist_kwargs = {"metric": "minkowski", "p": 3}
cvi_value = silhouette(X, clusters_pred, dist_kwargs=dist_kwargs)
print(f"Functional API    |  CVI value: {cvi_value:.4f}")

fout.close()
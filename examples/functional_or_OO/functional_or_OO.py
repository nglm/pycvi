import sys
out_fname = f'./output-functional_or_OO.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout
sys.path.append('./examples')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pycvi.cvi import Silhouette
from pycvi.cvi_func import silhouette
from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering

# -------------- Standard data handling operations ---------------------
# Load data
data, labels = load_data("xclara", "barton")

# Data pre-processing
scaler = StandardScaler()
X = scaler.fit_transform(data)

# ---------- Fit a clustering model and make predictions ---------------
# Assumed number of clusters
k = 3

# Train and predict a KMeans model
model = KMeans(n_clusters=k)
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
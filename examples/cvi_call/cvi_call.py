import sys
out_fname = f'./output-cvi_call.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout
sys.path.append('./examples')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pycvi.cvi import Silhouette
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

# ----------------------- Using CVI ------------------------------------
# Instanciate a CVI instance, could be any class defined in pycvi.cvi
cvi = Silhouette()

# Compute the CVI value of this clustering
cvi_value = cvi(X, clusters_pred)
print(f"Assumed number of clusters: {k}  |  CVI value: {cvi_value:.4f}")

fout.close()
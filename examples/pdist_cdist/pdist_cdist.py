import sys

out_fname = f'./output-pdist_cdist.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout

import numpy as np
from pycvi.datasets.benchmark import load_data
from pycvi.dist import f_cdist, f_pdist
from pycvi.cluster import get_clustering

# ===================== Non time series-data ===========================

dataset_name = "xclara"
data, labels = load_data(dataset_name, "barton")

# From predicted cluster-label for each datapoint to a list of
# datapoints for each cluster.
clustering = get_clustering(labels)
n_clusters = len(clustering)
print(f"\n{n_clusters} clusters in dataset {dataset_name} (Non time-series)")

# -------------------------- pdist -------------------------------------
# Distance matrices between the datapoints of each cluster
distances = [ f_pdist(data[cluster]) for cluster in clustering ]
for i, d in enumerate(distances):
    print(f"\nDistance matrix between datapoints in cluster {i+1}, shape {d.shape}")
    print(f"  Mean distance between datapoints in cluster {i+1}: {np.mean(d):.4f}")

# -------------------------- cdist -------------------------------------
# Distance matrices between the first cluster and the others
clusterA = clustering[0]
distances = [
    f_cdist(data[clusterA], data[cluster]) for cluster in clustering[1:]
]
for i, d in enumerate(distances):
    print(f"\nDistance matrix between cluster 1 and {i+2}, shape {d.shape}")
    print(f"  Mean distance between cluster 1 and {i+2}: {np.mean(d):.4f}")


# ======================= Time series-data =============================

dataset_name = "Trace"
data, labels = load_data(dataset_name, "ucr")

clustering = get_clustering(labels)
n_clusters = len(clustering)
print(f"\n{n_clusters} clusters in dataset {dataset_name} (Time-series)")

# -------------------------- pdist -------------------------------------
# Distance matrices between the datapoints of each cluster
distances = [ f_pdist(data[cluster]) for cluster in clustering ]
for i, d in enumerate(distances):
    print(f"\nDistance matrix between datapoints in cluster {i+1}, shape {d.shape}")
    print(f"  Mean distance between datapoints in cluster {i+1}: {np.mean(d):.4f}")

# -------------------------- cdist -------------------------------------
# Distance matrices between the first cluster and the others
clusterA = clustering[0]
distances = [
    f_cdist(data[clusterA], data[cluster]) for cluster in clustering[1:]
]
for i, d in enumerate(distances):
    print(f"\nDistance matrix between cluster 1 and {i+2}, shape {d.shape}")
    print(f"  Mean distance between cluster 1 and {i+2}: {np.mean(d):.4f}")

fout.close()
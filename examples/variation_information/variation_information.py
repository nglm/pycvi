import sys
sys.path.append('./examples')
out_fname = f'./output-variation_information_KMedoids.txt'
fout = open(out_fname, 'wt')
sys.stdout = fout

from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from pycvi.datasets.benchmark import load_data
from pycvi.cluster import get_clustering
from pycvi.vi import variation_information

from pycvi_examples_utils import plot_true_selected

# Load data
datasets = ["xclara", "zelnik1"]
for dataset in datasets:
    print(f" ============= {dataset} =============")
    data, labels = load_data(dataset, "barton")

    # Data pre-processing
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # --- Generate clusters assuming the correct number of clusters ----

    # From predicted cluster-label for each datapoint to a list of
    # datapoints for each cluster.
    clustering_true = get_clustering(labels)
    k_true = len(clustering_true)

    # Generate the clusters assuming the right number of clusters
    # Clustering model to use, could be any sklearn-like clustering class
    model = KMedoids(n_clusters=k_true)
    labels_pred = model.fit_predict(X)
    clustering_pred = get_clustering(labels_pred)

    # ------ variation of information between true and predicted -------

    # Compute the variation of information between the true clustering and
    # the clustering obtained with the method on the dataset.
    vi = variation_information(clustering_true, clustering_pred)
    print(f"Variation of information: {vi}")

    # ---------------------- Summmary fig ------------------------------
    ax_titles = [
        f"True clustering, k={k_true}",
        f"Clustering assuming k={k_true} | VI={vi:.4f}",
    ]
    fig = plot_true_selected(data, clustering_true, clustering_pred, ax_titles)
    fig_title = f"{dataset} - KMedoids clustering"
    fig_name = f"variation_information_KMedoids_{dataset}.png"
    fig.suptitle(fig_title)
    fig.savefig(fig_name)


fout.close()
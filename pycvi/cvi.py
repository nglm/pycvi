
from typing import List, Sequence, Union, Any, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

from .compute_scores import (
    compute_score, f_inertia, f_pdist, get_centroid, reduce, f_cdist,
)
from .cluster import (
    compute_center, generate_uniform, compute_cluster_params, prepare_data,
    get_clusters, sliding_window
)
from ._configuration import set_data_shape, get_model_parameters
from .utils import check_dims

def _clusters_from_uniform(X, n_clusters):
    """
    Helper function for "compute_gap_statistic"
    """
    N = len(X)

    # Fit a KMeans model to the sample from a uniform distribution
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    # Sort members into the different clusters and compute their
    # cluster info
    clusters = []
    for label_i in range(n_clusters):

        # Members belonging to that clusters
        members = [m for m in range(N) if labels[m] == label_i]
        if members == []:
            raise ValueError('No members in cluster')

        clusters.append(members)

    return clusters

def _compute_Wk(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
):
    """
    Helper function for some scores (gap, hartigan, CH, etc.)

    Pooled within-cluster sum of squares around the cluster means (WCSS)
    There are two ways to compute it, using distance to centroid or
    pairwise, we use pairwise, to avoid using barycenters.
    """
    dist_kwargs.setdefault("metric", "sqeuclidean")
    # Compute the log of the within-cluster dispersion of the clustering
    nis = [len(c) for c in clusters]
    d_intra = [f_pdist(X[c], dist_kwargs) for c in clusters]
    Wk = sum([intra/(2*ni) for (ni, intra) in zip(nis, d_intra)])
    return Wk

def _dist_between_centroids(
    X: np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
) -> List[float]:
    """
    Helper function for some scores (CH, score function, etc.)

    List of distances between cluster centroids and global centroid
    """
    global_center = np.expand_dims(compute_center(X), 0)
    dist = [
        # Distance between the centroids and the global centroid
        f_pdist(
            np.expand_dims(compute_center(X[c]), 0),
            global_center,
            dist_kwargs
        )
        for c in clusters
    ]
    return dist

def gap_statistic(
    X : np.ndarray,
    clusters: List[List[int]],
    B: int = 10
) -> float:
    """
    Compute the Gap statistics for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :param midpoint_w: center point of the time window w_t, used as
    reference in case of DTW, defaults to None.
    :type midpoint_w: int, optional
    :param B: Number of uniform samples drawn, defaults to 10
    :type B: int, optional
    :return: The gap statistics
    :rtype: float
    """
    k = len(clusters)

    # Compute the log of the within-cluster dispersion of the clustering
    wcss = np.log(_compute_Wk(X, clusters))

    # Generate B random datasets with the same shape as the input data
    random_datasets = [np.random.rand(X.shape) for _ in range(B)]

    # Compute the log of the within-cluster dispersion for each random dataset
    wcss_rand = []
    for X_rand in random_datasets:
        clusters_rand = _clusters_from_uniform(X_rand, k)
        wcss_rand.append(np.log(_compute_Wk(X_rand, clusters_rand)))

    # Compute the gap statistic for the current clustering
    gap = np.mean(wcss_rand) - wcss
    return gap

def score_function(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the score function for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of members
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The score function index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    dist_kwargs = {"metric" : 'euclidean'}

    nis = [len(c) for c in clusters]

    bdc = 1/(N*k) * sum(
        np.multiply(nis, _dist_between_centroids(X, clusters, dist_kwargs))
    )

    wdc = sum([
        f_inertia(c, dist_kwargs) / len(c)
        for c in clusters
    ])

    sf = 1 - (1 / (np.exp(np.exp(bdc - wdc))))

    return sf

def hartigan(
    X : np.ndarray,
    clustersk1: List[Tuple[List[int], Dict]],
    clustersk2: List[Tuple[List[int], Dict]],
) -> float:
    """
    Compute the hartigan index for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The hartigan index
    :rtype: float
    """
    N = len(X)
    k = len(clustersk1)
    if k == N:
        hartigan = 0
    else:
        Wk1 = _compute_Wk(X, clustersk1)
        Wk2 = _compute_Wk(X, clustersk2)

        hartigan = (Wk1/Wk2 - 1)*(N-k-1)

    return hartigan

def silhouette(
    X : np.ndarray,
    clusters: List[List[int]],
) -> float:
    """
    Compute the silhouette score for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The silhouette score
    :rtype: float
    """

    S_i1 = []
    nis = [len(c) for c in clusters]

    for i1, (c1, _) in enumerate(clusters):
        ni = len(c1)

        # Compute 'a' for all x (=X[m]) in c1 (excluding x in c1)
        a = [(1/nis[i1]) * reduce(f_cdist(X[c1], X[m]), "sum") for m in c1]

        # Compute 'b' for all x (=X[m]) in c1
        b = [np.min([
                reduce(f_cdist(X[c2], X[m]), "mean")
                for i2, (c2, _) in enumerate(clusters) if i1 != i2
            ]) for m in c1]

        # Silhouette score for cluster c1
        # The usual formula is written a sum divided by the number of x,
        # which is the mean
        S_i1.append(np.mean(
            [(b_x - a_x) / (max(b_x, a_x)) for (a_x, b_x) in zip(a, b)]
        ))

    # Silhouette score of the clustering
    # The usual formula is written a sum divided by k, which is the mean
    S = np.mean(S_i1)
    return S

def CH(
    X : np.ndarray,
    clusters: List[List[int]],
    dist_kwargs: dict = {},
    score_kwargs: dict = {},
) -> float:
    """
    Compute the Calinski–Harabasz (CH) index  for a given clustering

    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
    :param clusters: List of (members, info) tuples
    :type clusters: List[Tuple(List[int], Dict)]
    :return: The CH index
    :rtype: float
    """
    N = len(X)
    k = len(clusters)
    dist_kwargs.setdefault("metric", "sqeuclidean")

    # If we forget about the (N-k) / (k-1) factor, CH is defined and
    # is equal to 0
    if k == 1:
        CH = 0
    # Very special case for the case k=0
    elif k == 0:

        X0 = score_kwargs.get("X0", generate_uniform(X))
        #
        # Option 1: use the centroid of the uniform distribution
        # clusters0 = score_kwargs.get(
        #     "clusters0",
        #     [compute_cluster_params(X0, score_kwargs.get("midpoint_w", 0))]
        # )
        #
        # Option 2: use the original centroid
        #
        # The numerator can be seen as the distance between the global
        # centroid and N singletons uniformly distributed
        # Which can be seen as d(C0, c)
        sep = f_cdist(X0, np.expand_dims(compute_center(X), 0), dist_kwargs)

        # The denominator can be seen as the distance between the
        # original data and its centroid, which would correspond to the
        # denominator or the case k=1 (which is not used because CH(1) =
        # 0)
        # Note that the list has actually only one element
        comp = sum([
            f_inertia(X[c], dist_kwargs) for c in clusters
        ])
        CH = - N * (sep / comp)

    # Normal case
    else:
        nis = [len(c) for c in clusters]
        sep = sum(
            np.multiply(nis, _dist_between_centroids(X, clusters, dist_kwargs))
        )

        comp = sum([
            f_inertia(X[c], dist_kwargs) for c in clusters
        ])

        CH = (N-k) / (k-1) * (sep / comp)
    return CH


def generate_all_clusterings(
    data: np.ndarray,
    model_class,
    n_clusters_range: Sequence = None,
    DTW: bool = True,
    time_window: int = None,
    transformer = None,
    scaler = StandardScaler(),
    model_kw: dict = {},
    fit_predict_kw: dict = {},
    model_class_kw: dict = {},
    quiet: bool = True,
) -> List[List[List[int]]]:
    """
    Generate and return all clusterings.

    `clusters_t_n[t_w][k][i]` is a list of members indices contained in
    cluster i for the clustering assuming k clusters for the extracted
    time window t_w.

    :rtype: List[List[List[int]]]
    """
    # --------------------------------------------------------------
    # --------------------- Preliminary ----------------------------
    # --------------------------------------------------------------

    data_copy = set_data_shape(data)
    (N, T, d) = data_copy.shape

    if n_clusters_range is None:
        n_clusters_range = range(N)

    if time_window is not None:
        wind = sliding_window(T, time_window)
    else:
        wind = None

    if not DTW:
        # X: (N, d*T)
        data_copy = data_copy.reshape(N, d*T)

    # list of T (if sliding window) or 1 array(s) of shape:
    # (N, T|w_t, d) if DTW
    # (N, (T|w_t)*d) if not DTW
    data_clus = prepare_data(data_copy, DTW, wind, transformer, scaler)
    n_windows = len(data_clus)

    # --------------------------------------------------------------
    # --------------------- Find clusters --------------------------
    # --------------------------------------------------------------

    # temporary variable to help remember clusters before merging
    # clusters_t_n[t_w][k][i] is a list of members indices contained in
    # cluster i for the clustering assuming k clusters for the extracted
    # time window t_w
    clusters_t_n = [{} for _ in range(n_windows)]

    for t_w in n_windows:
        # ---- clustering method specific parameters --------
        # X_clus of shape (N, w_t*d) or (N, w_t, d)
        X_clus = data_clus[t_w]

        # Get clustering model parameters required by the
        # clustering model
        model_kw, fit_predict_kw, model_class_kw = get_model_parameters(
            model_class,
            model_kw = model_kw,
            fit_predict_kw = fit_predict_kw,
            model_class_kw = model_class_kw,
        )
        # Update fit_predict_kw with the data
        fit_predict_kw[model_class_kw["X_arg_name"]] = X_clus

        for n_clusters in n_clusters_range:

            # All members in the same cluster. Go to next iteration
            if n_clusters <= 1:
                clusters_t_n[t_w][n_clusters] = [[i for i in range(N)]]
            # General case
            else:

                # Update model_kw with the number of clusters
                model_kw[model_class_kw["k_arg_name"]] = n_clusters

                # ---------- Fit & predict using clustering model-------
                try :
                    clusters = get_clusters(
                        model_class,
                        model_kw = model_kw,
                        fit_predict_kw = fit_predict_kw,
                        model_class_kw = model_class_kw,
                    )
                except ValueError as ve:
                    if not quiet:
                        print(str(ve))
                    clusters_t_n[t_w][n_clusters] = None
                    continue

                clusters_t_n[t_w][n_clusters] = clusters

    return clusters_t_n


def compute_all_clustering_params(
    data: np.ndarray,
    model_class,
    n_clusters_range: Sequence = None,
    DTW: bool = True,
    model_kw: dict = {},
    fit_predict_kw: dict = {},
    model_class_kw: dict = {},
    quiet: bool = True,
) -> List[Tuple[List[int], dict]]:
    """
    Compute and return all scores

    :rtype: List[Tuple[List[int], dict]]
    """

def compute_all_scores(
    data: np.ndarray,
    clusterings: List[List[int]],
    DTW: bool = True,
    quiet: bool = True,
) -> List[float]:
    """
    Compute and return all scores

    :rtype: List[float]
    """

    # --------------------------------------------------------------
    # -------- Compute score, cluster params, etc. -----------------
    # --------------------------------------------------------------

    data_copy = set_data_shape(data)
    (N, T, d) = data_copy.shape
    data0 = generate_uniform(data_copy)

    if not DTW:
        # X: (N, d*T)
        data_copy = data_copy.reshape(N, d*T)
        data0 = data0.reshape(N, d*T)

    # Use default scaler and transformer
    # TODO: allow for a sliding window
    data_clus = prepare_data(data_copy)
    data_clus0 = prepare_data(data0)

    for n_clusters in clusterings.keys():
        # Take the data used for clustering while taking into account the
        # difference between time step indices with/without sliding window
        if n_clusters == 0:

            # X_clus: list of length T of arrays of shape (N, w_t, d)
            # X: (N, w_t, d)
            X_clus = data_clus0
        else:
            X_clus = data_clus

        if not DTW:
            # We take the entire time window into consideration for the
            # scores of the clusters
            # X: (N, d*T)
            X = X.reshape(N, T*d)
            # We take only the midpoint into consideration for the
            # parameters of the clusters
            # X_params: (N, d)
            X_params = X_params[:, midpoint_w, :]

        # Find cluster membership of each member
        clusters = clusters_n[n_clusters]

        # Go to next iteration if the model didn't converge
        if clusters is None:
            continue

        # -------- Cluster infos for each cluster ---------

        clusters = [
            (
                c,
                {
                    **{"k": [n_clusters]},
                    **compute_cluster_params(X_params[c], midpoint_w)
                }
            ) for c in clusters
        ]
        clusters_score = [
            (c, compute_cluster_params(X[c], midpoint_w))
            for c in clusters
        ]

        # ------------ Score corresponding to 'n_clusters' ---------
        score = compute_score(
            pg._score,
            X = X,
            clusters = clusters_score,
        )
        if n_clusters == 0:
            pg._zero_scores[t] = score
            # don't insert the case k=0 in cluster_data
            # nor in local_steps, go straight to the next iteration
        elif pg._score_type == "monotonous" and score :
            pass
        else:
            step_info = {"score" : score}

            # ---------------- Finalize local step -----------------
            # Find where should we insert this future local step
            idx = sort_fc(local_scores[t], score)
            local_scores[t].insert(idx, score)
            cluster_data[t].insert(idx, clusters)
            pg._local_steps[t].insert(
                idx,
                {**{'param' : {"k" : n_clusters}},
                    **step_info
                }
            )
            pg._nb_steps += 1
            pg._nb_local_steps[t] += 1

            if pg._verbose:
                print(" ========= ", t, " ========= ")
                msg = "n_clusters: " + str(n_clusters)
                for (key,item) in step_info.items():
                    msg += '  ||  ' + key + ":  " + str(item)
                print(msg)

    return cluster_data

def cvi(
    X: np.ndarray,
    cvi: Union[str, callable],
    clustering_model,
    k_range: Sequence = None,
) -> List[float]:

import numpy as np
from typing import List

def P_clusters(
    clustering: List[List[int]]
) -> List[float]:
    """
    List of probability of the outcome being in cluster i

    :param clustering: A given clustering
    :type clustering: List[List[int]]
    :return: List of probability of the outcome being in cluster i
    :rtype: List[float]
    """
    nis = [len(c) for c in clustering]
    N = int(np.sum(nis))
    return [ni/N for ni in nis]

def entropy(
    clustering: List[List[int]]
) -> float:
    """
    Entropy of the given clustering

    :param clustering: A given clustering
    :type clustering: List[List[int]]
    :return: Entropy of the given clustering
    :rtype: float
    """
    P_ks = P_clusters(clustering)
    return - float(np.sum(
        [P_k * np.log10(P_k) for P_k in P_ks]
    ))

def contingency_matrix(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
) -> np.ndarray:
    """
    Contingency matrix between two clusterings.

    :param clustering1: First clustering
    :type clustering1: List[List[int]]
    :param clustering2: Second clustering
    :type clustering2: List[List[int]]
    :return: Contingency matrix between the two clusterings.
    :rtype: np.ndarray
    """
    k1 = len(clustering1)
    k2 = len(clustering2)
    return np.array([
            [len(list(set(c1).intersection(c2))) for c2 in clustering2]
            for c1 in clustering1
        ], dtype=int)

def mutual_information(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
) -> float:
    """
    Mutual information between two clusterings.

    :param clustering1: First clustering
    :type clustering1: List[List[int]]
    :param clustering2: Second clustering
    :type clustering2: List[List[int]]
    :return: Mutual information between two clusterings.
    :rtype: float
    """
    m = contingency_matrix(clustering1, clustering2)
    P_ks1 = P_clusters(clustering1)
    P_ks2 = P_clusters(clustering2)
    tmp = [
        [
            m[i1, i2] * np.log10( m[i1, i2]/ (P_k1 * P_k2 ))
            if m[i1, i2]>0 else 0
            for i2, P_k2 in enumerate(P_ks2)
        ] for i1, P_k1 in enumerate(P_ks1)
    ]
    I =  float(np.sum(tmp))
    return I

def variational_information(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
) -> float:
    """
    Variational information between two clusterings.

    :param clustering1: First clustering
    :type clustering1: List[List[int]]
    :param clustering2: Second clustering
    :type clustering2: List[List[int]]
    :return: Variational information between two clusterings.
    :rtype: float
    """
    H1 = entropy(clustering1)
    H2 = entropy(clustering2)
    I = mutual_information(clustering1, clustering2)
    return H1 + H2 - 2*I
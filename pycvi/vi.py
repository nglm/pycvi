import numpy as np
from typing import List, Tuple

from .exceptions import InvalidKError

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

    Conventions: (see "Elements of Information Theory" by Cover and
    Thomas), section 2.3

    - log is in base 2 and entropy is count in bits
    - $0 log(0/0) = 0$
    - $0 log(0/q) = 0$
    - $p log(p/0) = +inf$

    :param clustering: A given clustering
    :type clustering: List[List[int]]
    :return: Entropy of the given clustering
    :rtype: float
    """
    P_ks = P_clusters(clustering)
    return - float(np.sum(
        [P_k * np.log2(P_k) if P_k>0 else 0 for P_k in P_ks ]
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
    N = sum([len(c) for c in clustering1])
    m = np.array([
            [len(list(set(c1).intersection(c2)))/N for c2 in clustering2]
            for c1 in clustering1
        ])
    return m

def mutual_information(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
) -> float:
    """
    Mutual information between two clusterings.

    Conventions: (see "Elements of Information Theory" by Cover and
    Thomas), section 2.3

    - log is in base 2 and entropy is count in bits
    - $0 log(0/0) = 0$
    - $0 log(0/q) = 0$
    - $p log(p/0) = +inf$

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
            m[i1, i2] * np.log2( m[i1, i2]/ (P_k1 * P_k2 ))
            if m[i1, i2]>0 else 0
            for i2, P_k2 in enumerate(P_ks2)
        ] for i1, P_k1 in enumerate(P_ks1)
    ]
    I =  float(np.sum(tmp))
    return I

def variation_information(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
) -> float:
    """
    Variational information between two clusterings.

    :param clustering1: First clustering
    :type clustering1: List[List[int]]
    :param clustering2: Second clustering
    :type clustering2: List[List[int]]
    :return: Variation of information between two clusterings.
    :rtype: float
    """
    H1 = entropy(clustering1)
    H2 = entropy(clustering2)
    I = mutual_information(clustering1, clustering2)
    return H1 + H2 - 2*I

def _align_clusterings(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Align `clustering2` to `clustering1`.

    To be aligned the clusterings must have the same number of clusters

    :param clustering1: First clustering, used as reference
    :type clustering1: List[List[int]]
    :param clustering2: Second clustering, to be aligned
    :type clustering2: List[List[int]]
    :return: Same clusters but "aligned" to `clustering1`
    :rtype: Tuple[List[List[int]], List[List[int]]]
    """
    if len(clustering1) != len(clustering2):
        msg = (
            "clustering1 and clustering2 can't be aligned because their"
            + "lengths don't match: {} and {}."
        ).format(len(clustering1), len(clustering2))
        raise InvalidKError(msg)

    # Make a safe copy of clustering2 where we will delete one by one
    # clusters that are already aligned
    left_c2 = [c.copy() for c in clustering2]
    sorted_c1 = sorted(clustering1, key=len, reverse=True)
    res_c2 = []

    # While not all clusters have been processed, take the biggest
    # cluster in c1
    for c1 in sorted_c1:

        # Find the cluster in clustering 2 that has the largest common
        # datapoints with the largest cluster in c1
        argbest = np.argmax([
            len(list(set(c1).intersection(c2))) for c2 in left_c2
        ])

        # Add to the result and remove from left_c2
        res_c2.append(left_c2[argbest].copy())
        del left_c2[argbest]

    return sorted_c1, res_c2

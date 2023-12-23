"""
PyCVI exceptions.
"""

class EmptyClusterError(ValueError):
    """
    The clustering algorithm didn't converge.

    As a result of not converging, some clusters are empty resulting
    in the absence of clustering.
    """
    pass

class ShapeError(ValueError):
    """
    The given data doesn't have the right shape.

    The acceptable input shapes are: `(N,)` or `(N, d)` or `(N, T, d)`.
    """
    pass

class InvalidScoreError(ValueError):
    """
    The score given is not among the implemented scores.

    Either because its reduction was not implemented or because its main
    function was not implemented.
    """
    pass

class ScoreError(ValueError):
    """
    The score given is invalid, probably because it is `None`.
    """
    pass

class InvalidKError(ValueError):
    """
    The CVI was called with an incompatible number of clusters.
    """
    pass

class SelectionError(ValueError):
    """
    The CVI couldn't select the best clustering.

    This is probably because the CVI values given are None.
    """
    pass
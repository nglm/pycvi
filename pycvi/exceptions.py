class NoClusterError(ValueError):
    """
    The clustering algorithm didn't converge and some clusters are empty
    resulting in the absence of clustering.
    """
    pass

class ShapeError(ValueError):
    """
    The given data doesn't have the right shape:
    `(N,)` or `(N, d)` or `(N, T, d)`
    """
    pass

class InvalidScoreError(ValueError):
    """
    The score given is invalid. Either because its reduction was not
    implemented or because its main function was not implemented.
    """
    pass

class ScoreError(ValueError):
    """
    The score given is invalid, probably because it is `None`
    """
    pass

class InvalidKError(ValueError):
    """
    The CVI was called with an incompatible number of clusters, or
    two clusterings are to be aligned but their k don't match
    """
    pass

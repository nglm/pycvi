import numpy as np

def check_dims(a, ndim):
    dims = a.shape
    if (len(dims) == ndim - 1):
        a = np.expand_dims(a, 0)
    elif len(dims) != ndim:
        msg = f"Array should have {ndim} dimensions, but has shape {dims}"
        raise ValueError(msg)
    return a

def match_dims(a1, a2):
    dims1 = a1.shape
    dims2 = a2.shape
    if (len(dims1) == len(dims2)):
        return a1, a2
    elif (len(dims1) == len(dims2) - 1):
        a = np.expand_dims(a1, 0)
        return a, a2
    elif (len(dims2) == len(dims1) - 1):
        a2 = np.expand_dims(a2, 0)
        return a1, a
    else:
        msg = f"Cannot make dimensions {dims1} and {dims2} match."
        raise ValueError(msg)
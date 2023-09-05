import numpy as np
import pandas as pd
import urllib
from scipy.io import arff
import io

def check_dims(a, ndim):
    dims = a.shape
    if (len(dims) == ndim - 1):
        a = np.expand_dims(a, 0)
    elif len(dims) != ndim:
        msg = f"Array should have {ndim} dimensions, but has shape {dims}"
        raise ValueError(msg)
    return a

def match_dims(a1, a2):
    """
    Make a1 and a2 have the same shape, assuming that there might
    be a dimension of size 1 missing in front of one of the arrays
    """
    dims1 = a1.shape
    dims2 = a2.shape
    # If there are already matching just return the original arrays
    if (len(dims1) == len(dims2)):
        return a1, a2
    # If a1 misses a dim
    elif (len(dims1) == len(dims2) - 1):
        a = np.expand_dims(a1, 0)
        return a, a2
    # If a2 misses a dim
    elif (len(dims2) == len(dims1) - 1):
        a2 = np.expand_dims(a2, 0)
        return a1, a2
    else:
        msg = f"Cannot make dimensions {dims1} and {dims2} match."
        raise ValueError(msg)

def load_data_from_github(url):
    ftpstream = urllib.request.urlopen(url)
    data, meta = arff.loadarff(io.StringIO(ftpstream.read().decode('utf-8')))
    df = pd.DataFrame(data)
    # Get only data, not the labels and convert to numpy
    data = df.iloc[:, 0:-1].to_numpy()
    return data, meta
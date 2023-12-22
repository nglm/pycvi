import numpy as np
import pandas as pd
import urllib.request
from scipy.io import arff
import io
from typing import List, Dict, Union, Tuple

from .exceptions import ShapeError

def _check_dims(a, ndim):
    dims = a.shape
    if (len(dims) == ndim - 1):
        a = np.expand_dims(a, 0)
    elif len(dims) != ndim:
        msg = f"Array should have {ndim} dimensions, but has shape {dims}"
        raise ShapeError(msg)
    return a

def _match_dims(a1, a2):
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
        raise ShapeError(msg)

def _load_data_from_github(url):
    ftpstream = urllib.request.urlopen(url)
    data, meta = arff.loadarff(io.StringIO(ftpstream.read().decode('utf-8')))
    df = pd.DataFrame(data)
    # Get only data, not the labels and convert to numpy
    data = df.iloc[:, 0:-1].to_numpy()
    return data, meta

def _check_list_of_dict(
    l: Union[List[Dict], Dict]
) -> Tuple[List[Dict], bool]:
    """
    Outputs a List[Dict] from either a Dict or a List[Dict].

    List[Dict] is for all variables involving time-series with sliding
    window. All concerned PyCVI functions assume the input to be
    a List[Dict] but a Dict is also acceptable if the dataset was not
    time series, or time-series to be clustered considering all time
    steps at once.

    Parameters
    ----------
    l : Union[List[Dict], Dict]
        The input variable, it could be clusterings, CVI values or
        selected k.

    Returns
    -------
    Tuple[List[Dict], bool]
        The same input but as a List[Dict] and a boolean indicating
        whether the input was only a list or not.

    Raises
    ------
    ValueError
        If ```l``` is empty or not of the right type (a
        list of dictionaries in the case of time series data
        clustered by sliding windows or a dictionary).
    """
    return_list = True
    if type(l) == dict:
        if l == {}:
            msg = (
                "input must be non-empty"
            )
            raise ValueError(msg)
        return_list = False
        l_of_dict = [l]
    elif type(l) == list:
        if l == []:
            msg = (
                "input must be non-empty"
            )
            raise ValueError(msg)
        elif type(l[0]) != dict:
            msg = (
                "input must be of type "
                + f"List[dict] or dict. Got List[{type(l[0])}] instead"
            )
            raise ValueError(msg)
        return_list = True
        l_of_dict = l
    else:
        msg = (
            "input must be of type List[dict] or dict "
            + f"Got {type(l)} instead."
        )
        raise ValueError(msg)
    return l_of_dict, return_list
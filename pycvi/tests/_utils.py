import numpy as np

def _aux_check_float(value, or_None=False):
    if or_None:
        assert (type(value) in [float, np.float64, type(None)]), type(value)
    else:
        assert (type(value) in [float, np.float64]), type(value)
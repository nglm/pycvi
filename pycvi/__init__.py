# We typically want:
#
# import pycvi as cvi
# from pycvi import gap_statistics
#
# gap = cvi.gap_statistics(X, clusters_data)

# We do that to skip the cvi file and access the functions directly)
from .cvi import gap_statistics
from .scores import SCORES

from . import scores
from . import cvi
from . import cluster
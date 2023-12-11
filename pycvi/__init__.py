# We typically want:
#
# import pycvi as cvi
# from pycvi import gap_statistic
#
# gap = cvi.gap_statistic(X, clusters_data)

# We do that to skip the cvi file and access the functions directly)
from .scores import SCORES

from . import scores
from . import cvi
from . import cluster
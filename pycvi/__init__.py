"""
Internal Cluster Validity Indices (CVIs), compatible with DTW and DBA.
"""

# We typically want:
#
# import pycvi as cvi
# from pycvi import gap_statistic
#
# gap = cvi.gap_statistic(X, clusters_data)

# We do that to skip the cvi file and access the functions directly)
from .cvi import CVIs

from . import cvi
from . import cvi_func
from . import cluster
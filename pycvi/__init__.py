# We typically want:
#
# import pycvi as cvi
# from pycvi import gap_statistics
#
# gap = cvi.gap_statistics(X, clusters_data)

# We do that to skip the cvi file and access the functions directly)
from .cvi import gap_statistics

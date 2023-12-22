cdist and pdist
------------------

In this example, we use the PyCVI counterparts of `pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_ and `cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_: in scipy, namely :func:`pycvi.dist.f_pdist`  and :func:`pycvi.dist.f_cdist` in order to compute distance matrices with time-series data, the same way these functions are used with non-time series data. Their behavior is the same as scipy's functions but for time-series data, DTW is used a a distance function (`aeon <https://www.aeon-toolkit.org/en/stable/>`_'s implementation is used).

Note that in the case of :func:`pycvi.dist.f_pdist`, a condensed distance matrix is returned (as in scipy).

.. literalinclude:: ../../examples/pdist_cdist/pdist_cdist.py
   :lines: 7-65

.. literalinclude:: ../../examples/pdist_cdist/output-pdist_cdist.txt
   :language: text
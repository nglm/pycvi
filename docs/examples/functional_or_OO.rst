Functional and Object-oriented APIs
------------------------------------

All implemented CVIs take as mandatory input a dataset ``X`` and a clustering ``clusters``. In addition, all implemented CVIs take as optional parameter a dictionary of keyword arguments ``dist_kwargs`` for the distance function used to compute pairwise distances between datapoints.

- If the dataset ``X`` is time-series data and if DTW is used, the distance function is based on `aeon.distances.dtw_pairwise_distance <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.dtw_pairwise_distance.html#dtw-pairwise-distance>`_. In that case, the ``dist_kwargs`` keyword argument can include parameters such as ``window`` or ``itakura_max_slope``.
- Otherwise, the distance function used is based on `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_. In that case, ``dist_kwargs`` can define the same parameters as this function.

In addition, some CVI functions take additional optional parameters, which can be specified when using the ``__call__`` method of the corresponding CVI class via the ``cvi_kwargs`` keyword argument. Below is an example of the correspondance between the functional API (:func:`pycvi.cvi_func.silhouette`) and the object-oriented API (:class:`pycvi.cvi.Silhouette`) for the silhouette CVI, but the same principle applies to all CVIs.

.. literalinclude:: ../../examples/functional_or_OO/functional_or_OO.py
   :lines: 7-44
   :emphasize-lines: 30-32,36-37

.. literalinclude:: ../../examples/functional_or_OO/output-functional_or_OO.txt
   :language: text
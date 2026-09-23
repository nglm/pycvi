Functional and Object-oriented APIs
------------------------------------

All implemented CVIs take a dataset ``X`` and a clustering ``clusters`` as
mandatory inputs. They also accept the optional dictionary ``dist_kwargs``
for the distance function used to compute pairwise distances between data
points.

- If the dataset ``X`` is time-series data and if ``ts_dist=True``, the distance function is based on `aeon.distances.pairwise_distance <https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.pairwise_distance.html#pairwise-distance>`_. In that case, the ``dist_kwargs`` keyword argument can include parameters such as ``method``, ``window`` or ``itakura_max_slope``.
- Otherwise, the distance function used is based on `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_. In that case, ``dist_kwargs`` can define the same parameters as this function.

Some CVI functions take additional optional parameters. These can be supplied
to the ``__call__`` method of the corresponding CVI class through the
``cvi_kwargs`` keyword argument. The example below shows the correspondence
between the functional API (:func:`pycvi.cvi_func.silhouette`) and the
object-oriented API (:class:`pycvi.cvi.Silhouette`) for the Silhouette CVI;
the same principle applies to all CVIs.

.. literalinclude:: ../../examples/functional_or_OO/functional_or_OO.py
   :linenos:
   :start-after: sys.stdout = fout
   :end-before: fout.close()
   :emphasize-lines: 31-33,37-38

.. literalinclude:: ../../examples/functional_or_OO/output-functional_or_OO.txt
   :language: text
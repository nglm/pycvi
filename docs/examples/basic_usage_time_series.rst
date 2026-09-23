CVI - Basic usage with time-series
----------------------------------

In this example, we integrate PyCVI into a clustering pipeline with
time-series data to select the best clustering. We try several values of
:math:`k`, the number of clusters, and pass each value to
``TimeSeriesKMeans(n_clusters=k)``. Here, :math:`k` is the main parameter of
the clustering method. The example uses MSM as the distance measure and MBA
as the cluster center, both designed for time-series data.

.. include:: /examples/examples_reminders.rst

.. literalinclude:: ../../examples/basic_usage_time_series/basic_usage_time_series.py
   :linenos:
   :start-after: sys.stdout = fout
   :end-before: fout.close()
   :emphasize-lines: 20, 31-41, 48

.. image:: ../../examples/basic_usage_time_series/basic_usage_TS_KMeans_Dunn.png

.. literalinclude:: ../../examples/basic_usage_time_series/output-basic_usage_TS_KMeans_Dunn.txt
   :language: text
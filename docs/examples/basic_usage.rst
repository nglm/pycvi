CVI - Basic usage
------------------

In this example, we integrate PyCVI into a usual clustering pipeline to
select the best clustering. We try several values of :math:`k`, the number
of clusters, and pass each value to ``KMeans(n_clusters=k)``. Here, :math:`k`
is the main parameter of the clustering method.

.. include:: /examples/examples_reminders.rst

.. literalinclude:: ../../examples/basic_usage/basic_usage.py
   :linenos:
   :start-after: sys.stdout = fout
   :end-before: fout.close()
   :emphasize-lines: 18, 29-39, 46

.. image:: ../../examples/basic_usage/basic_usage_KMeans_Silhouette.png

.. literalinclude:: ../../examples/basic_usage/output-basic_usage_KMeans_Silhouette.txt
   :language: text
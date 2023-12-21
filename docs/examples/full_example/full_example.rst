Full example
-------------

Here is an example using exclusively PyCVI for the entire clustering pipeline. The preprocessing steps and the clustering steps can be integrated into the PyCVI pipeline by providing sklearn-like classes of clustering models (e.g. `KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_) and data preprocessor (e.g. `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_).

In this example, we use time-series data and non-time-series data. In addition we use classes from `scikit-learn <https://scikit-learn.org/stable/index.html>`_, `scikit-learn extra <https://scikit-learn-extra.readthedocs.io/en/stable/>`_ and `aeon <https://www.aeon-toolkit.org/en/latest/index.html>`_ in order to illustrate the compatibility of PyCVI with sklearn-like libraries.

.. literalinclude:: ../../examples/full_example/full_example.py

.. image:: ../../examples/full_example/Barton_data_KMeans.png

.. image:: ../../examples/full_example/Barton_data_AgglomerativeClustering_Single.png

.. image:: ../../examples/full_example/UCR_data_no_DTW_KMedoids.png

.. image:: ../../examples/full_example/UCR_data_DTW_TimeSeriesKMeans.png

.. literalinclude:: ../../examples/full_example/output-full_example.txt
   :language: text
Using the Variation of Information
------------------------------------

In this example, we compute the variation of information between the true clustering and the clustering predicted when assuming the correct number of clusters. We see that some clusterings methods are not adapted to some dataset and this is illustrated by a high variation of information (VI) between the predicted and the true clustering.

.. include:: /examples/examples_reminders.rst

.. literalinclude:: ../../examples/variation_information/variation_information.py
   :linenos:
   :start-after: sys.stdout = fout
   :end-before: fout.close()
   :emphasize-lines: 29-31, 37

.. image:: ../../examples/variation_information/variation_information_KMeans_xclara.png

.. image:: ../../examples/variation_information/variation_information_KMeans_zelnik1.png

.. literalinclude:: ../../examples/variation_information/output-variation_information_KMeans.txt
   :language: text
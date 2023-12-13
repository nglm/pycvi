.. PyCVI documentation master file, created by
   sphinx-quickstart on Sun Dec 10 11:51:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. mdinclude:: md/welcome_top.md

.. toctree::
   :maxdepth: 4
   :caption: Tutorials:

   md/tutorials.md

Main Modules
----------------------------

All implemented CVIs are available here:

.. autosummary::
   :toctree:
   :caption: Main Features

   pycvi.cvi

Additional functions are defined to compute clusterings, compare clusterings and evaluate clusterings:

.. autosummary::
   :toctree:
   :template: function.rst

   pycvi.cluster.generate_all_clusterings
   pycvi.vi.variation_information
   pycvi.compute_scores.compute_all_scores


.. toctree::
   :maxdepth: 3
   :caption: Full API:

   api

.. mdinclude:: md/welcome_bottom.md

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

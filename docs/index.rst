.. PyCVI documentation master file, created by
   sphinx-quickstart on Sun Dec 10 11:51:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Main welcoming text
   ---------------------------------------

.. mdinclude:: md/welcome_top.md

.. Tutorials sections
   ---------------------------------------

.. .. toctree::
..    :maxdepth: 4
..    :caption: Tutorials:

..    md/tutorials.md

.. Tutorials sections
   ---------------------------------------

.. toctree::
   :maxdepth: 4
   :caption: Examples:

   examples/examples

.. Main module section
   ---------------------------------------

Main Modules
----------------------------

All implemented CVIs are available here:

.. autosummary::
   :toctree:
   :nosignatures:
   :caption: Main Features

   pycvi.cvi

High level functions are defined to compute clusterings, compare clusterings and evaluate clusterings:

.. autosummary::
   :toctree:
   :nosignatures:
   :template: function.rst

   pycvi.cluster.generate_all_clusterings
   pycvi.vi.variation_information
   pycvi.compute_scores.compute_all_scores

More low-level functions are defined to perform common operations but that can handle the case of DTW and DBA if working on time-series data:

.. autosummary::
   :nosignatures:
   :template: function.rst

   pycvi.compute_scores.f_pdist
   pycvi.compute_scores.f_cdist
   pycvi.compute_scores.f_inertia
   pycvi.cluster.compute_center

.. Full API section
   ---------------------------------------

.. include:: api.rst

.. Contribute / Issues, etc.
   ---------------------------------------

.. mdinclude:: md/welcome_bottom.md

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

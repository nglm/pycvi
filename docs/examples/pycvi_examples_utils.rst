Utils functions for example scripts
=====================================

All the examples scripts showcased in this documentation produce some figures. The code to generate these figures have nothing to do with ``PyCVI``, so we decided to define plot functions in a separate file so that only PyCVI related features are emphasized in each example. On this page, you can find the code source of the plot functions that are necessary in order to run the examples scripts.

This file has to be copied and saved in the same directory as your examples scripts. Alternatively, you can copy the utils functions directly into your scripts (and remove the line starting with ``from pycvi_examples_utils import``).

.. include:: /examples/examples_reminders.rst

.. literalinclude:: ../../examples/pycvi_examples_utils.py
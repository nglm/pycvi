Running example scripts on your computer
==========================================

The examples given in this documentation are mainly meant to be used as an inspiration for your own scripts, and to showcase ``PyCVI`` features. But it is of course also possible to run these examples directly on your computer. To do so, please follow the extra steps detailed on this page.

Extra dependencies
--------------------

In order to run the example scripts, extra dependencies are necessary. The install command is then:

.. code-block:: bash

    # For poetry
    poetry add pycvi-lib[examples]
    # For pip and anaconda
    pip install pycvi-lib[examples]

Alternatively, you can manually install in your environment the packages that are necessary to run the example scripts (``matplotlib`` and/or ``scikit-learn-extra`` depending on the example).

Utils functions
--------------------

In addition to extra dependencies, utils functions are imported in each example script from the ``pycvi_examples_utils.py`` file using a line starting with ``from pycvi_examples_utils import ...``. This file and can be found on this documentation :doc:`here </examples/pycvi_examples_utils>` or directly in the `source repository <https://github.com/nglm/pycvi/blob/master/examples/pycvi_examples_utils.py>`_.

If you wish to run the examples, please copy the content of the ``pycvi_examples_utils.py`` file and save it on the same directory level as your example script. Alternatively, you can copy the utils functions directly into your scripts (and remove the line starting with ``from pycvi_examples_utils import``).

Running your scripts
---------------------

Once you have done the steps above, you can run your scripts (for example ``basic_usage.py``) in your python environment with ``PyCVI`` and the extra dependencies installed by running the following command:

.. code-block:: bash

    # If you run from the same directory as your scripts
    python basic_usage.py
    # Alternatively, if you run from another directory as your scripts'
    python path/to/your/script/basic_usage.py
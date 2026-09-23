Running example scripts on your computer
==========================================

The examples in this documentation are mainly intended as inspiration for
your own scripts and to showcase ``PyCVI`` features. You can also run them
directly on your computer by following the extra steps detailed on this page.

Extra dependencies
--------------------

In order to run the example scripts, extra dependencies are necessary. The install command is then:

.. code-block:: bash

    # for uv
    uv add pycvi-lib --optional examples
    # For poetry
    poetry add pycvi-lib -E examples
    # For pip and anaconda
    pip install pycvi-lib[examples]

Alternatively, you can manually install in your environment the packages that are necessary to run the example scripts (``matplotlib`` and/or ``scikit-learn-extra`` depending on the example).

.. include:: ../md/warning_dependencies.md
   :parser: myst_parser.sphinx_

Utility functions
--------------------

In addition to these dependencies, each example imports utility functions
from ``pycvi_examples_utils.py`` with a line beginning
``from pycvi_examples_utils import ...``. The file is available in this
documentation :doc:`here </examples/pycvi_examples_utils>` and in the
`source repository <https://github.com/nglm/pycvi/blob/master/examples/pycvi_examples_utils.py>`_.

To run the examples, copy the contents of ``pycvi_examples_utils.py`` into
the same directory as your example script. Alternatively, copy the utility
functions directly into your script and remove the import line beginning
``from pycvi_examples_utils import``.

Running your scripts
---------------------

Once you have completed these steps, run a script (for example
``basic_usage.py``) from a Python environment with ``PyCVI`` and the extra
dependencies installed:

.. code-block:: bash

    # If you run from the same directory as your scripts
    python basic_usage.py
    # Alternatively, if you run from another directory as your scripts'
    python path/to/your/script/basic_usage.py
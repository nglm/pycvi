# PyCVI 0.1.2 release notes

2024/07/02

## Python versions

This version supports Python versions 3.8 to 3.11.

## Fixes

- Some dependency incompatibilities have been fixed.

## Features

- Example scripts can now be run with the set of extra dependencies "`examples`". This extra set of dependency can be installed by running:

  ```bash
  # using poetry
  poetry add pycvi-lib[examples]
  # using pip or anaconda
  pip install pycvi-lib[examples]
  ```

- All instructions to run examples scripts are now available in the documentation: [Running example scripts on your computer](https://pycvi.readthedocs.io/en/latest/examples/examples_instructions.html).

## Contributors

- Natacha Galmiche (@nglm)
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
# Because we created the sphinx project in the ./doc folder
sys.path.insert(0, os.path.abspath('..'))
# Because tslearn is not among the standard libraries
autodoc_mock_imports = ["tslearn"]


# -- Project information -----------------------------------------------------

project = 'PyCVI'
copyright = '2023, Natacha Galmiche'
author = 'Natacha Galmiche'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Note that as of December 2023, if 'm2r' and 'myst_parser'
# can not be activated simultaneously, however m2r seems to include
# myst_parser
extensions = [
    # To generate documentation from docstrings
    'sphinx.ext.autodoc',
    # To be able to use numpy styles or google styles of docstrings
    'sphinx.ext.napoleon',
    # To get only the first line of all functions and potentially use
    # it as a toctree
    'sphinx.ext.autosummary',
    # To tell sphinx that we are also using markdown
    # 'myst_parser',
    # To be able to include md files directly in rst files
    # adds the mdinclude directive
    'm2r',
]

# Uncomment if m2r is NOT included in the extension
# But myst_parser IS
# To tell sphinx to look also for .md files.
# source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']